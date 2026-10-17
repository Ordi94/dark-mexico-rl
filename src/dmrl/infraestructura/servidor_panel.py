"""
Servidor local del panel: sirve la pagina estatica y transmite cuadros por WebSocket.

Reglas de diseno:
    * Solo escucha en la interfaz de bucle local (127.0.0.1); cualquier otra direccion se rechaza.
    * publicar() nunca espera: sin navegadores conectados no serializa nada y con alguno lento
      descarta los cuadros mas viejos de SU cola. Cada navegador tiene su propio hilo de escritura,
      de modo que uno atascado no detiene a los demas ni a la pelea.
    * Se valida Host y Origin para que una pagina cualquiera abierta en el navegador del usuario no
      pueda leer el flujo de la pelea (ni resolver un nombre externo hacia 127.0.0.1).
    * Las peticiones se acotan en tamano, metodo y ruta (lista de admitidos).
"""

from __future__ import annotations

import contextlib
import json
import select
import socket
import threading
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

from dmrl.infraestructura import websocket_minimo as ws

HOST_LOCAL = "127.0.0.1"
PUERTO_DEFECTO = 8765
CARPETA_WEB = Path(__file__).parent / "panel_web"
RUTAS_PAGINA = ("/", "/index.html")
RUTA_WEBSOCKET = "/ws"
RUTA_SALUD = "/salud"
MAXIMO_CABECERA = 8192
MAXIMO_CLIENTES = 8
MAXIMO_CONEXIONES = 32
COLA_POR_CLIENTE = 4
PLAZO_PETICION = 5.0
PLAZO_ENVIO = 5.0
ESPERA_LECTURA = 0.5

ProveedorMensajes = Callable[[], list[dict[str, Any]]]


class _Peticion:
    """
    Peticion HTTP ya leida y acotada.

    Contexto:
        ServidorPanel._atender.

    Atributos:
        metodo: GET, HEAD, etc., en mayusculas.
        ruta: ruta sin parametros de consulta.
        cabeceras: nombres en minusculas.
    """

    def __init__(self, metodo: str, ruta: str, cabeceras: dict[str, str]) -> None:
        """
        Guarda los campos de la peticion.

        Contexto:
            _leer_peticion.

        Args:
            metodo (str): Metodo HTTP.
            ruta (str): Ruta pedida.
            cabeceras (dict[str, str]): Cabeceras con nombre en minusculas.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            _Peticion("GET", "/", {})
        """
        self.metodo = metodo
        self.ruta = ruta
        self.cabeceras = cabeceras


def _leer_peticion(conexion: socket.socket) -> _Peticion | None:
    """
    Lee y analiza la cabecera de una peticion HTTP.

    Contexto:
        ServidorPanel._atender. Todo lo recibido es hostil: se limita el tamano y se descarta lo que
        no tenga la forma esperada.

    Args:
        conexion (socket.socket): Socket recien aceptado, con plazo.

    Returns:
        _Peticion | None: La peticion, o None si esta mal formada, es demasiado grande o vence el plazo.

    Raises:
        Ninguna: Los errores de red devuelven None.

    Ejemplo:
        peticion = _leer_peticion(conexion)
    """
    datos = b""
    try:
        while b"\r\n\r\n" not in datos:
            trozo = conexion.recv(1024)
            if not trozo or len(datos) + len(trozo) > MAXIMO_CABECERA:
                return None
            datos += trozo
    except OSError:
        return None
    lineas = datos.split(b"\r\n\r\n", 1)[0].decode("latin-1").split("\r\n")
    partes = lineas[0].split(" ")
    if len(partes) != 3 or not partes[2].startswith("HTTP/1."):
        return None
    cabeceras: dict[str, str] = {}
    for linea in lineas[1:]:
        nombre, separador, valor = linea.partition(":")
        if not separador:
            return None
        cabeceras[nombre.strip().lower()] = valor.strip()
    return _Peticion(partes[0].upper(), partes[1].split("?", 1)[0], cabeceras)


def _respuesta(estado: str, cuerpo: bytes = b"", tipo: str = "text/plain; charset=utf-8",
               extra: dict[str, str] | None = None) -> bytes:
    """
    Arma una respuesta HTTP con cabeceras de seguridad.

    Contexto:
        ServidorPanel; todas las respuestas cierran la conexion.

    Args:
        estado (str): Linea de estado, por ejemplo "200 OK".
        cuerpo (bytes): Contenido.
        tipo (str): Content-Type.
        extra (dict[str, str], opcional): Cabeceras adicionales.

    Returns:
        bytes: Respuesta completa.

    Raises:
        Ninguna.

    Ejemplo:
        _respuesta("404 Not Found", b"no")
    """
    cabeceras = {"Content-Type": tipo, "Content-Length": str(len(cuerpo)), "Connection": "close",
                 "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"}
    cabeceras.update(extra or {})
    texto = "".join(f"{k}: {v}\r\n" for k, v in cabeceras.items())
    return f"HTTP/1.1 {estado}\r\n{texto}\r\n".encode("latin-1") + cuerpo


class _ConexionWs:
    """
    Un navegador conectado: cola propia de cuadros y hilo de escritura.

    Contexto:
        ServidorPanel. La cola acotada descarta los cuadros mas viejos cuando el navegador no
        alcanza el ritmo, en lugar de acumular memoria o frenar a quien publica.
    """

    def __init__(self, conexion: socket.socket) -> None:
        """
        Prepara la cola y el cierre de la conexion.

        Contexto:
            Tras un apreton de manos valido.

        Args:
            conexion (socket.socket): Socket ya actualizado a WebSocket.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            _ConexionWs(conexion)
        """
        self._conexion = conexion
        self._cola: deque[bytes] = deque(maxlen=COLA_POR_CLIENTE)
        self._aviso = threading.Event()
        self._envio = threading.Lock()
        self.abierta = True
        self.descartados = 0

    def encolar(self, trama: bytes) -> None:
        """
        Agrega una trama sin esperar; si la cola esta llena se pierde la mas vieja.

        Contexto:
            ServidorPanel.publicar, desde el hilo de la pelea.

        Args:
            trama (bytes): Trama ya codificada.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            cliente.encolar(trama)
        """
        if len(self._cola) == self._cola.maxlen:
            self.descartados += 1
        self._cola.append(trama)
        self._aviso.set()

    def enviar_ahora(self, trama: bytes) -> bool:
        """
        Envia una trama de forma sincrona desde el hilo que llama.

        Contexto:
            Mensajes iniciales, pong y cierre; no se usa desde el hilo de la pelea.

        Args:
            trama (bytes): Trama ya codificada.

        Returns:
            bool: True si se envio; False si la conexion fallo (queda cerrada).

        Raises:
            Ninguna.

        Ejemplo:
            cliente.enviar_ahora(trama)
        """
        try:
            with self._envio:
                self._conexion.sendall(trama)
            return True
        except OSError:
            self.cerrar()
            return False

    def bombear(self) -> None:
        """
        Bucle del hilo de escritura: vacia la cola hacia el navegador hasta que se cierre.

        Contexto:
            Un hilo por navegador.

        Args:
            Ninguno.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            threading.Thread(target=cliente.bombear, daemon=True).start()
        """
        while self.abierta:
            self._aviso.wait(ESPERA_LECTURA)
            self._aviso.clear()
            while self.abierta:
                try:
                    trama = self._cola.popleft()
                except IndexError:
                    break
                if not self.enviar_ahora(trama):
                    return

    def cerrar(self) -> None:
        """
        Cierra la conexion y despierta al hilo de escritura.

        Contexto:
            Fin de la sesion, error de red o parada del servidor.

        Args:
            Ninguno.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna: Los errores al cerrar se ignoran.

        Ejemplo:
            cliente.cerrar()
        """
        self.abierta = False
        self._aviso.set()
        with contextlib.suppress(OSError):
            self._conexion.close()


class ServidorPanel:
    """
    Servidor HTTP + WebSocket del panel, en hilos propios.

    Contexto:
        Lo crea panel_vivo.iniciar_sesion_panel. El hilo de la pelea solo llama a publicar().
    """

    def __init__(self, puerto: int = PUERTO_DEFECTO, host: str = HOST_LOCAL,
                 mensajes_iniciales: ProveedorMensajes | None = None, pagina: Path | None = None) -> None:
        """
        Configura el servidor sin abrir el puerto.

        Contexto:
            Arranque del panel.

        Args:
            puerto (int): Puerto TCP; 0 deja que el sistema elija uno libre (pruebas).
            host (str): Direccion de escucha; solo se admite 127.0.0.1.
            mensajes_iniciales (Callable, opcional): Devuelve los mensajes que recibe cada navegador
                al conectarse (configuracion e historial).
            pagina (Path, opcional): HTML a servir; por defecto panel_web/index.html.

        Returns:
            None: No retorna valor.

        Raises:
            ValueError: Si host no es la interfaz de bucle local o el puerto esta fuera de 0..65535.

        Ejemplo:
            servidor = ServidorPanel(8765)
        """
        if host != HOST_LOCAL:
            raise ValueError(f"el panel solo puede escuchar en {HOST_LOCAL}, no en {host}")
        if not 0 <= puerto <= 65535:
            raise ValueError(f"puerto fuera de 0..65535: {puerto}")
        self._host = host
        self._puerto = puerto
        self._proveedor = mensajes_iniciales
        self._pagina = pagina or CARPETA_WEB / "index.html"
        self._escucha: socket.socket | None = None
        self._clientes: tuple[_ConexionWs, ...] = ()
        self._candado = threading.Lock()
        self._detenido = threading.Event()
        self._cupo = threading.BoundedSemaphore(MAXIMO_CONEXIONES)
        self.cuadros_publicados = 0
        self.cuadros_no_serializables = 0

    def fijar_proveedor(self, proveedor: ProveedorMensajes) -> None:
        """
        Define quien aporta los mensajes iniciales de cada navegador.

        Contexto:
            El proveedor (la politica del panel) se crea despues del servidor porque necesita su
            metodo publicar; se conecta antes de abrir el puerto.

        Args:
            proveedor (Callable): Devuelve la lista de mensajes a enviar al conectarse.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            servidor.fijar_proveedor(politica.mensajes_iniciales)
        """
        self._proveedor = proveedor

    @property
    def puerto(self) -> int:
        """
        Puerto en que escucha (el real si se pidio el 0).

        Contexto:
            Mensajes al usuario y pruebas.

        Args:
            Ninguno.

        Returns:
            int: Puerto TCP.

        Raises:
            Ninguna.

        Ejemplo:
            servidor.puerto
        """
        return self._puerto

    @property
    def url(self) -> str:
        """
        Direccion de la pagina del panel.

        Contexto:
            Abrir el navegador y mensajes al usuario.

        Args:
            Ninguno.

        Returns:
            str: URL http con el puerto real.

        Raises:
            Ninguna.

        Ejemplo:
            servidor.url  # "http://127.0.0.1:8765/"
        """
        return f"http://{self._host}:{self._puerto}/"

    @property
    def clientes(self) -> int:
        """
        Navegadores conectados por WebSocket en este momento.

        Contexto:
            Pruebas y diagnostico.

        Args:
            Ninguno.

        Returns:
            int: Cantidad de conexiones abiertas.

        Raises:
            Ninguna.

        Ejemplo:
            servidor.clientes
        """
        return len(self._clientes)

    @property
    def cuadros_descartados(self) -> int:
        """
        Cuadros perdidos en las colas de navegadores lentos (acumulado de los conectados).

        Contexto:
            Diagnostico del rendimiento.

        Args:
            Ninguno.

        Returns:
            int: Cuadros descartados.

        Raises:
            Ninguna.

        Ejemplo:
            servidor.cuadros_descartados
        """
        return sum(c.descartados for c in self._clientes)

    def iniciar(self) -> int:
        """
        Abre el puerto y lanza el hilo que acepta conexiones.

        Contexto:
            Antes de abrir la ventana del juego, para que el navegador ya pueda conectarse.

        Args:
            Ninguno.

        Returns:
            int: Puerto real en escucha.

        Raises:
            OSError: Si el puerto esta ocupado o no se puede abrir.
            RuntimeError: Si ya estaba iniciado.

        Ejemplo:
            puerto = ServidorPanel(8765).iniciar()
        """
        if self._escucha is not None:
            raise RuntimeError("el servidor del panel ya esta iniciado")
        escucha = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            escucha.bind((self._host, self._puerto))
            escucha.listen(MAXIMO_CONEXIONES)
        except OSError:
            escucha.close()
            raise
        escucha.settimeout(ESPERA_LECTURA)
        self._escucha = escucha
        self._puerto = int(escucha.getsockname()[1])
        threading.Thread(target=self._aceptar, name="panel-aceptar", daemon=True).start()
        return self._puerto

    def detener(self) -> None:
        """
        Cierra el puerto y todas las conexiones.

        Contexto:
            Fin de la pelea.

        Args:
            Ninguno.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna: Los errores al cerrar se ignoran.

        Ejemplo:
            servidor.detener()
        """
        self._detenido.set()
        if self._escucha is not None:
            with contextlib.suppress(OSError):
                self._escucha.close()
        with self._candado:
            clientes, self._clientes = self._clientes, ()
        for cliente in clientes:
            cliente.cerrar()

    def publicar(self, mensaje: dict[str, Any]) -> None:
        """
        Envia un mensaje a todos los navegadores sin esperar nunca.

        Contexto:
            PoliticaPanel, desde el hilo de la pelea, una vez por decision. Sin navegadores no
            serializa; un navegador lento pierde cuadros viejos; un mensaje no serializable se descarta.

        Args:
            mensaje (dict[str, Any]): Mensaje JSON (camelCase).

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            servidor.publicar({"tipo": "cuadro", "paso": 1})
        """
        clientes = self._clientes
        if not clientes or self._detenido.is_set():
            return
        try:
            carga = json.dumps(mensaje, separators=(",", ":"), allow_nan=False).encode("utf-8")
            trama = ws.codificar_trama(ws.OPCODE_TEXTO, carga)
        except (TypeError, ValueError):
            self.cuadros_no_serializables += 1
            return
        self.cuadros_publicados += 1
        for cliente in clientes:
            cliente.encolar(trama)

    def _aceptar(self) -> None:
        """
        Bucle del hilo que acepta conexiones y lanza un hilo por cada una.

        Contexto:
            Hilo "panel-aceptar".

        Args:
            Ninguno.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            (hilo interno)
        """
        escucha = self._escucha
        assert escucha is not None
        while not self._detenido.is_set():
            try:
                conexion, _ = escucha.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            if not self._cupo.acquire(blocking=False):
                conexion.close()
                continue
            threading.Thread(target=self._atender, args=(conexion,), name="panel-conexion", daemon=True).start()

    def _atender(self, conexion: socket.socket) -> None:
        """
        Atiende una conexion: pagina, salud o WebSocket.

        Contexto:
            Un hilo por conexion; libera su cupo al terminar.

        Args:
            conexion (socket.socket): Socket aceptado.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna: Cualquier error cierra la conexion.

        Ejemplo:
            (hilo interno)
        """
        try:
            conexion.settimeout(PLAZO_PETICION)
            peticion = _leer_peticion(conexion)
            if peticion is None:
                conexion.sendall(_respuesta("400 Bad Request", b"peticion no valida"))
            elif (rechazo := self._rechazar(peticion)) is not None:
                conexion.sendall(rechazo)
            elif peticion.ruta == RUTA_WEBSOCKET:
                self._websocket(conexion, peticion)
            else:
                conexion.sendall(self._pagina_o_salud(peticion))
        except OSError:
            pass
        finally:
            with contextlib.suppress(OSError):
                conexion.close()
            self._cupo.release()

    def _rechazar(self, peticion: _Peticion) -> bytes | None:
        """
        Respuesta de rechazo si la peticion no cumple las reglas, o None si se acepta.

        Contexto:
            _atender. Lista de admitidos: metodo GET, rutas conocidas, Host y Origin locales.

        Args:
            peticion (_Peticion): Peticion analizada.

        Returns:
            bytes | None: Respuesta HTTP de error, o None.

        Raises:
            Ninguna.

        Ejemplo:
            self._rechazar(peticion)
        """
        locales = {f"{HOST_LOCAL}:{self._puerto}", f"localhost:{self._puerto}"}
        if peticion.metodo != "GET":
            return _respuesta("405 Method Not Allowed", b"solo GET", extra={"Allow": "GET"})
        if peticion.cabeceras.get("host", "") not in locales:
            return _respuesta("403 Forbidden", b"host no admitido")
        origen = peticion.cabeceras.get("origin")
        if origen is not None and origen not in {f"http://{h}" for h in locales}:
            return _respuesta("403 Forbidden", b"origen no admitido")
        if peticion.ruta not in (*RUTAS_PAGINA, RUTA_WEBSOCKET, RUTA_SALUD):
            return _respuesta("404 Not Found", b"no existe")
        return None

    def _pagina_o_salud(self, peticion: _Peticion) -> bytes:
        """
        Respuesta para la pagina del panel o el estado del servidor.

        Contexto:
            _atender, tras superar _rechazar.

        Args:
            peticion (_Peticion): Peticion admitida que no es WebSocket.

        Returns:
            bytes: Respuesta HTTP.

        Raises:
            Ninguna: Si falta la pagina se responde 500.

        Ejemplo:
            self._pagina_o_salud(peticion)
        """
        if peticion.ruta == RUTA_SALUD:
            estado = {"clientes": self.clientes, "cuadrosPublicados": self.cuadros_publicados,
                      "cuadrosDescartados": self.cuadros_descartados}
            return _respuesta("200 OK", json.dumps(estado).encode("utf-8"), "application/json")
        try:
            html = self._pagina.read_bytes()
        except OSError:
            return _respuesta("500 Internal Server Error", b"falta la pagina del panel")
        puerto = self._puerto
        politica = ("default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
                    f"connect-src ws://{HOST_LOCAL}:{puerto} ws://localhost:{puerto}; img-src data:")
        return _respuesta("200 OK", html, "text/html; charset=utf-8", {"Content-Security-Policy": politica})

    def _websocket(self, conexion: socket.socket, peticion: _Peticion) -> None:
        """
        Apreton de manos WebSocket, mensajes iniciales y bucle de lectura.

        Contexto:
            _atender, para la ruta /ws.

        Args:
            conexion (socket.socket): Socket aceptado.
            peticion (_Peticion): Peticion de actualizacion ya admitida.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna: Los errores de red terminan la conexion.

        Ejemplo:
            (uso interno)
        """
        cabeceras = peticion.cabeceras
        clave = cabeceras.get("sec-websocket-key", "")
        valida = (cabeceras.get("upgrade", "").lower() == "websocket"
                  and "upgrade" in cabeceras.get("connection", "").lower()
                  and cabeceras.get("sec-websocket-version") == "13" and ws.clave_valida(clave))
        if not valida:
            conexion.sendall(_respuesta("400 Bad Request", b"actualizacion WebSocket no valida"))
            return
        if self.clientes >= MAXIMO_CLIENTES:
            conexion.sendall(_respuesta("503 Service Unavailable", b"demasiados navegadores conectados"))
            return
        conexion.sendall((f"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                          f"Sec-WebSocket-Accept: {ws.clave_aceptacion(clave)}\r\n\r\n").encode("ascii"))
        cliente = _ConexionWs(conexion)
        conexion.settimeout(PLAZO_ENVIO)
        for mensaje in self._proveedor() if self._proveedor else []:
            if not cliente.enviar_ahora(self._trama(mensaje)):
                return
        with self._candado:
            self._clientes = (*self._clientes, cliente)
        threading.Thread(target=cliente.bombear, name="panel-escritura", daemon=True).start()
        try:
            self._leer_navegador(conexion, cliente)
        finally:
            with self._candado:
                self._clientes = tuple(c for c in self._clientes if c is not cliente)
            cliente.cerrar()

    @staticmethod
    def _trama(mensaje: dict[str, Any]) -> bytes:
        """
        Codifica un mensaje JSON como trama de texto.

        Contexto:
            Mensajes iniciales de cada navegador.

        Args:
            mensaje (dict[str, Any]): Mensaje JSON.

        Returns:
            bytes: Trama lista para enviar.

        Raises:
            ValueError: Si el mensaje contiene NaN o infinitos.
            TypeError: Si el mensaje no es serializable.

        Ejemplo:
            ServidorPanel._trama({"tipo": "configuracion"})
        """
        carga = json.dumps(mensaje, separators=(",", ":"), allow_nan=False).encode("utf-8")
        return ws.codificar_trama(ws.OPCODE_TEXTO, carga)

    def _leer_navegador(self, conexion: socket.socket, cliente: _ConexionWs) -> None:
        """
        Atiende las tramas del navegador (ping, cierre) hasta que se cierre la conexion.

        Contexto:
            _websocket. El navegador no envia datos utiles; lo que llegue se lee solo para
            contestar pings y detectar el cierre. Una trama sin mascara, fragmentada o demasiado
            grande cierra la conexion.

        Args:
            conexion (socket.socket): Socket del navegador.
            cliente (_ConexionWs): Conexion registrada.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna: Los errores de red y de protocolo terminan el bucle.

        Ejemplo:
            (uso interno)
        """
        while cliente.abierta and not self._detenido.is_set():
            try:
                listo, _, _ = select.select([conexion], [], [], ESPERA_LECTURA)
                if not listo:
                    continue
                opcode, carga, enmascarada = ws.leer_trama(conexion, maximo=125)
            except (OSError, ws.ProtocoloError):
                return
            if not enmascarada or opcode == ws.OPCODE_CIERRE:
                cliente.enviar_ahora(ws.codificar_trama(ws.OPCODE_CIERRE, b""))
                return
            if opcode == ws.OPCODE_PING:
                cliente.enviar_ahora(ws.codificar_trama(ws.OPCODE_PONG, carga))


def puerto_libre(puerto: int, host: str = HOST_LOCAL) -> bool:
    """
    Indica si un puerto TCP local puede abrirse ahora.

    Contexto:
        Comprobacion previa al arranque del panel, antes de lanzar el juego.

    Args:
        puerto (int): Puerto a probar.
        host (str): Direccion local.

    Returns:
        bool: True si se pudo reservar un instante.

    Raises:
        Ninguna.

    Ejemplo:
        puerto_libre(8765)
    """
    sonda = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sonda.bind((host, puerto))
        return True
    except OSError:
        return False
    finally:
        sonda.close()

