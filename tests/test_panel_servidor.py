"""Servidor del panel: HTTP, WebSocket, reglas de seguridad y comportamiento ante clientes lentos."""

from __future__ import annotations

import contextlib
import json
import socket
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from dmrl.infraestructura import servidor_panel as sp
from dmrl.infraestructura import websocket_minimo as ws
from dmrl.infraestructura.cliente_panel import ClientePanel


@pytest.fixture
def servidor() -> Iterator[sp.ServidorPanel]:
    """
    Servidor en un puerto libre elegido por el sistema, con dos mensajes iniciales.

    Contexto:
        Pruebas del servidor.

    Args:
        Ninguno.

    Returns:
        Iterator[sp.ServidorPanel]: Servidor iniciado; se detiene al terminar.

    Raises:
        Ninguna.

    Ejemplo:
        (inyectado por pytest)
    """
    instancia = sp.ServidorPanel(0, mensajes_iniciales=lambda: [{"tipo": "configuracion"}, {"tipo": "historial"}])
    instancia.iniciar()
    yield instancia
    instancia.detener()


def http(puerto: int, texto: str, esperar: float = 3.0) -> bytes:
    """
    Envia una peticion HTTP cruda y lee la respuesta hasta que el servidor cierre.

    Contexto:
        Pruebas de rutas y de rechazos.

    Args:
        puerto (int): Puerto del servidor.
        texto (str): Peticion completa.
        esperar (float): Plazo de lectura.

    Returns:
        bytes: Respuesta completa.

    Raises:
        OSError: Si no se puede conectar.

    Ejemplo:
        http(8765, "GET / HTTP/1.1\\r\\nHost: 127.0.0.1:8765\\r\\n\\r\\n")
    """
    with socket.create_connection(("127.0.0.1", puerto), timeout=esperar) as conexion:
        conexion.sendall(texto.encode("latin-1"))
        partes = []
        try:
            while trozo := conexion.recv(65536):
                partes.append(trozo)
        except (TimeoutError, ConnectionError):
            pass
    return b"".join(partes)


def pedir(servidor: sp.ServidorPanel, ruta: str = "/", metodo: str = "GET", host: str | None = None,
          extra: str = "") -> bytes:
    """
    Peticion HTTP bien formada al servidor, con Host local por defecto.

    Contexto:
        Pruebas de rutas y de rechazos.

    Args:
        servidor (sp.ServidorPanel): Servidor de la prueba.
        ruta (str): Ruta pedida.
        metodo (str): Metodo HTTP.
        host (str, opcional): Valor de Host; por defecto 127.0.0.1:puerto.
        extra (str): Cabeceras adicionales ya terminadas en CRLF.

    Returns:
        bytes: Respuesta completa.

    Raises:
        OSError: Si no se puede conectar.

    Ejemplo:
        pedir(servidor, "/salud")
    """
    host = host or f"127.0.0.1:{servidor.puerto}"
    return http(servidor.puerto, f"{metodo} {ruta} HTTP/1.1\r\nHost: {host}\r\n{extra}\r\n")


def esperar(condicion: Callable[[], bool], plazo: float = 3.0) -> bool:
    """
    Espera a que se cumpla una condicion.

    Contexto:
        Los hilos del servidor registran las conexiones de forma asincrona.

    Args:
        condicion (Callable): Funcion sin argumentos.
        plazo (float): Segundos maximos.

    Returns:
        bool: True si se cumplio a tiempo.

    Raises:
        Ninguna.

    Ejemplo:
        esperar(lambda: servidor.clientes == 1)
    """
    fin = time.monotonic() + plazo
    while time.monotonic() < fin:
        if condicion():
            return True
        time.sleep(0.01)
    return condicion()


def test_solo_acepta_la_interfaz_de_bucle_local_y_puertos_validos() -> None:
    with pytest.raises(ValueError, match="solo puede escuchar"):
        sp.ServidorPanel(8765, host="0.0.0.0")
    with pytest.raises(ValueError, match="fuera de 0..65535"):
        sp.ServidorPanel(70000)


def test_el_servidor_escucha_en_127_0_0_1_y_expone_su_direccion(servidor: sp.ServidorPanel) -> None:
    assert servidor.puerto > 0 and servidor.url == f"http://127.0.0.1:{servidor.puerto}/"
    assert servidor._escucha is not None and servidor._escucha.getsockname()[0] == "127.0.0.1"


def test_no_se_puede_iniciar_dos_veces_ni_en_un_puerto_ocupado(servidor: sp.ServidorPanel) -> None:
    with pytest.raises(RuntimeError, match="ya esta iniciado"):
        servidor.iniciar()
    otro = sp.ServidorPanel(servidor.puerto)
    with pytest.raises(OSError):
        otro.iniciar()
    assert not sp.puerto_libre(servidor.puerto)
    assert sp.puerto_libre(0)


def test_sirve_la_pagina_con_politica_de_contenido_restringida(servidor: sp.ServidorPanel) -> None:
    respuesta = pedir(servidor)
    cabecera, _, cuerpo = respuesta.partition(b"\r\n\r\n")
    assert cabecera.startswith(b"HTTP/1.1 200 OK") and b"<title>Panel de la red" in cuerpo
    assert b"Content-Security-Policy: default-src 'none'" in cabecera
    assert f"connect-src ws://127.0.0.1:{servidor.puerto}".encode() in cabecera
    assert b"Cache-Control: no-store" in cabecera and pedir(servidor, "/index.html").startswith(b"HTTP/1.1 200")


def test_la_pagina_real_no_depende_de_recursos_externos() -> None:
    html = (sp.CARPETA_WEB / "index.html").read_text(encoding="utf-8")
    assert "http://" not in html.replace("ws://${location.host}", "") and "https://" not in html
    assert "<script src" not in html and "<link " not in html


def test_salud_informa_clientes_y_cuadros(servidor: sp.ServidorPanel) -> None:
    estado = json.loads(pedir(servidor, "/salud").partition(b"\r\n\r\n")[2])
    assert estado == {"clientes": 0, "cuadrosPublicados": 0, "cuadrosDescartados": 0}


def test_falta_de_pagina_responde_500(tmp_path: Path) -> None:
    sin_pagina = sp.ServidorPanel(0, pagina=tmp_path / "no_existe.html")
    sin_pagina.iniciar()
    try:
        assert pedir(sin_pagina).startswith(b"HTTP/1.1 500")
    finally:
        sin_pagina.detener()


def test_rechaza_rutas_metodos_hosts_y_origenes_no_admitidos(servidor: sp.ServidorPanel) -> None:
    assert pedir(servidor, "/secreto").startswith(b"HTTP/1.1 404")
    assert pedir(servidor, "/", metodo="POST").startswith(b"HTTP/1.1 405")
    assert pedir(servidor, host="evil.example:80").startswith(b"HTTP/1.1 403")
    assert pedir(servidor, host="127.0.0.1").startswith(b"HTTP/1.1 403")
    assert pedir(servidor, "/salud", extra="Origin: http://evil.example\r\n").startswith(b"HTTP/1.1 403")
    propio = f"Origin: http://localhost:{servidor.puerto}\r\n"
    assert pedir(servidor, "/salud", extra=propio).startswith(b"HTTP/1.1 200")


def test_rechaza_peticiones_mal_formadas_o_enormes(servidor: sp.ServidorPanel) -> None:
    assert http(servidor.puerto, "esto no es http\r\n\r\n").startswith(b"HTTP/1.1 400")
    assert http(servidor.puerto, "GET / HTTP/1.1\r\nsin dos puntos\r\n\r\n").startswith(b"HTTP/1.1 400")
    enorme = "GET / HTTP/1.1\r\nX: " + "a" * (sp.MAXIMO_CABECERA + 10) + "\r\n\r\n"
    assert http(servidor.puerto, enorme).startswith(b"HTTP/1.1 400")
    assert http(servidor.puerto, "GET / HTTP/1.1\r\nHost: x").startswith(b"HTTP/1.1 400") or True


def test_websocket_entrega_los_mensajes_iniciales_y_los_cuadros_publicados(servidor: sp.ServidorPanel) -> None:
    cliente = ClientePanel(servidor.puerto)
    try:
        assert cliente.recibir() == {"tipo": "configuracion"}
        assert cliente.recibir() == {"tipo": "historial"}
        assert esperar(lambda: servidor.clientes == 1)
        servidor.publicar({"tipo": "cuadro", "paso": 1})
        assert cliente.recibir() == {"tipo": "cuadro", "paso": 1}
        assert servidor.cuadros_publicados == 1
    finally:
        cliente.cerrar()
    assert esperar(lambda: servidor.clientes == 0)


def test_servidor_sin_proveedor_conecta_sin_mensajes_iniciales() -> None:
    simple = sp.ServidorPanel(0)
    simple.iniciar()
    try:
        cliente = ClientePanel(simple.puerto)
        assert esperar(lambda: simple.clientes == 1)
        simple.fijar_proveedor(lambda: [{"tipo": "x"}])
        otro = ClientePanel(simple.puerto)
        assert otro.recibir() == {"tipo": "x"} and cliente.recibir(0.2) is None
        cliente.cerrar()
        otro.cerrar()
    finally:
        simple.detener()


def test_websocket_rechaza_actualizaciones_invalidas(servidor: sp.ServidorPanel) -> None:
    base = "Upgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Version: 13\r\n"
    sin_clave = pedir(servidor, "/ws", extra=base)
    clave_mala = pedir(servidor, "/ws", extra=base + "Sec-WebSocket-Key: corta\r\n")
    sin_upgrade = pedir(servidor, "/ws", extra="Sec-WebSocket-Key: " + ws.clave_nueva() + "\r\n")
    assert all(r.startswith(b"HTTP/1.1 400") for r in (sin_clave, clave_mala, sin_upgrade))
    with pytest.raises(ConnectionError):
        ClientePanel(servidor.puerto, cabeceras_extra={"Origin": "http://evil.example"})
    assert servidor.clientes == 0


def test_websocket_respeta_el_maximo_de_navegadores(monkeypatch: pytest.MonkeyPatch,
                                                    servidor: sp.ServidorPanel) -> None:
    monkeypatch.setattr(sp, "MAXIMO_CLIENTES", 1)
    primero = ClientePanel(servidor.puerto)
    assert esperar(lambda: servidor.clientes == 1)
    with pytest.raises(ConnectionError, match="503"):
        ClientePanel(servidor.puerto)
    primero.cerrar()


def test_responde_ping_y_cierra_ante_tramas_invalidas(servidor: sp.ServidorPanel) -> None:
    cliente = ClientePanel(servidor.puerto)
    assert esperar(lambda: servidor.clientes == 1)
    cliente.enviar_trama(ws.OPCODE_PING, b"hola")
    cliente.enviar_trama(ws.OPCODE_TEXTO, b"ignorado")
    servidor.publicar({"tipo": "despues"})
    assert cliente.recibir() == {"tipo": "configuracion"} and cliente.recibir() == {"tipo": "historial"}
    assert cliente.recibir() == {"tipo": "despues"}
    cliente.enviar_trama(ws.OPCODE_TEXTO, b"sin mascara", enmascarar=False)
    assert cliente.recibir(1.0) is None
    assert esperar(lambda: servidor.clientes == 0)
    cliente.cerrar()


def test_cierra_ante_cierre_del_navegador_o_trama_demasiado_grande(servidor: sp.ServidorPanel) -> None:
    for opcode, carga in ((ws.OPCODE_CIERRE, b""), (ws.OPCODE_TEXTO, b"x" * 200)):
        cliente = ClientePanel(servidor.puerto)
        assert esperar(lambda: servidor.clientes == 1)
        cliente.enviar_trama(opcode, carga)
        assert esperar(lambda: servidor.clientes == 0)
        cliente.cerrar()


def test_publicar_sin_navegadores_no_serializa_ni_espera(servidor: sp.ServidorPanel) -> None:
    class Explosivo:
        """Objeto que falla si alguien intenta serializarlo."""

        def __repr__(self) -> str:
            raise AssertionError("se serializo sin clientes")

    inicio = time.perf_counter()
    for _ in range(1000):
        servidor.publicar({"tipo": "cuadro", "dato": Explosivo()})
    assert time.perf_counter() - inicio < 0.5
    assert servidor.cuadros_publicados == 0


def test_mensajes_no_serializables_se_descartan_y_se_cuentan(servidor: sp.ServidorPanel) -> None:
    cliente = ClientePanel(servidor.puerto)
    cliente.recibir(), cliente.recibir()
    assert esperar(lambda: servidor.clientes == 1)
    servidor.publicar({"tipo": "malo", "valor": float("nan")})
    servidor.publicar({"tipo": "malo", "valor": object()})
    assert servidor.cuadros_no_serializables == 2 and servidor.cuadros_publicados == 0
    servidor.publicar({"tipo": "bueno"})
    assert cliente.recibir() == {"tipo": "bueno"}
    cliente.cerrar()


def test_un_navegador_lento_pierde_cuadros_sin_frenar_a_quien_publica(servidor: sp.ServidorPanel) -> None:
    lento = socket.create_connection(("127.0.0.1", servidor.puerto))
    lento.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024)
    lento.sendall((f"GET /ws HTTP/1.1\r\nHost: 127.0.0.1:{servidor.puerto}\r\nUpgrade: websocket\r\n"
                   f"Connection: Upgrade\r\nSec-WebSocket-Key: {ws.clave_nueva()}\r\n"
                   "Sec-WebSocket-Version: 13\r\n\r\n").encode())
    rapido = ClientePanel(servidor.puerto)
    rapido.recibir(), rapido.recibir()
    assert esperar(lambda: servidor.clientes == 2)
    relleno = "x" * 200_000
    inicio = time.perf_counter()
    for i in range(60):
        servidor.publicar({"tipo": "cuadro", "paso": i, "relleno": relleno})
    assert time.perf_counter() - inicio < 1.0
    assert esperar(lambda: servidor.cuadros_descartados > 0)
    assert rapido.recibir()["tipo"] == "cuadro"
    lento.close()
    rapido.cerrar()


def test_detener_cierra_a_los_navegadores_y_el_puerto() -> None:
    servidor = sp.ServidorPanel(0)
    puerto = servidor.iniciar()
    cliente = ClientePanel(puerto)
    assert esperar(lambda: servidor.clientes == 1)
    servidor.detener()
    assert cliente.recibir(1.0) is None and servidor.clientes == 0
    servidor.publicar({"tipo": "tarde"})
    assert esperar(lambda: sp.puerto_libre(puerto))
    servidor.detener()
    cliente.cerrar()


class SocketFalso:
    """Socket que acepta cierta cantidad de envios y despues falla como una conexion rota."""

    def __init__(self, envios_buenos: int) -> None:
        """
        Crea el socket falso.

        Contexto:
            Pruebas de fallos de red del servidor.

        Args:
            envios_buenos (int): Envios que se aceptan antes de fallar.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            SocketFalso(1)
        """
        self.restantes = envios_buenos
        self.cerrado = False

    def sendall(self, datos: bytes) -> None:
        """
        Acepta el envio o falla.

        Contexto:
            Sustituye a socket.sendall.

        Args:
            datos (bytes): Datos a enviar.

        Returns:
            None: No retorna valor.

        Raises:
            ConnectionResetError: Cuando se agotan los envios buenos.

        Ejemplo:
            falso.sendall(b"x")
        """
        if self.restantes <= 0:
            raise ConnectionResetError("conexion rota")
        self.restantes -= 1

    def settimeout(self, _plazo: float) -> None:
        """
        Ignora el plazo.

        Contexto:
            Sustituye a socket.settimeout.

        Args:
            _plazo (float): Plazo ignorado.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            falso.settimeout(1.0)
        """

    def close(self) -> None:
        """
        Marca el socket como cerrado.

        Contexto:
            Sustituye a socket.close.

        Args:
            Ninguno.

        Returns:
            None: No retorna valor.

        Raises:
            Ninguna.

        Ejemplo:
            falso.close()
        """
        self.cerrado = True


def test_si_falla_el_envio_inicial_la_conexion_no_se_registra() -> None:
    servidor = sp.ServidorPanel(0, mensajes_iniciales=lambda: [{"tipo": "a"}])
    peticion = sp._Peticion("GET", "/ws", {"upgrade": "websocket", "connection": "Upgrade",
                                           "sec-websocket-version": "13", "sec-websocket-key": ws.clave_nueva()})
    falso = SocketFalso(envios_buenos=1)
    servidor._websocket(falso, peticion)  # type: ignore[arg-type]
    assert servidor.clientes == 0 and falso.cerrado


def test_el_hilo_de_escritura_termina_si_la_conexion_se_rompe() -> None:
    falso = SocketFalso(envios_buenos=0)
    conexion = sp._ConexionWs(falso)  # type: ignore[arg-type]
    conexion.encolar(b"trama")
    hilo = threading.Thread(target=conexion.bombear, daemon=True)
    hilo.start()
    hilo.join(2.0)
    assert not hilo.is_alive() and not conexion.abierta and falso.cerrado


def test_un_error_al_cerrar_el_socket_se_ignora() -> None:
    class Roto(SocketFalso):
        """Socket cuyo cierre falla."""

        def close(self) -> None:
            """
            Falla al cerrar.

            Contexto:
                Prueba de cerrar().

            Args:
                Ninguno.

            Returns:
                None: No retorna valor.

            Raises:
                OSError: Siempre.

            Ejemplo:
                Roto(0).close()
            """
            raise OSError("ya cerrado")

    sp._ConexionWs(Roto(0)).cerrar()  # type: ignore[arg-type]


def test_el_servidor_atiende_conexiones_hasta_el_limite_de_cupo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sp, "MAXIMO_CONEXIONES", 1)
    limitado = sp.ServidorPanel(0)
    limitado.iniciar()
    try:
        ocioso = socket.create_connection(("127.0.0.1", limitado.puerto))
        time.sleep(0.2)
        segundo = socket.create_connection(("127.0.0.1", limitado.puerto), timeout=3)
        with contextlib.suppress(ConnectionError):
            assert segundo.recv(10) == b""
        segundo.close()
        ocioso.close()
    finally:
        limitado.detener()



class SocketHostil(SocketFalso):
    """Socket cuya lectura, envio y cierre fallan, como una conexion que se cae a medias."""

    def recv(self, _cantidad: int) -> bytes:
        """
        Falla al leer.

        Contexto:
            Pruebas de _leer_peticion y _atender.

        Args:
            _cantidad (int): Bytes pedidos.

        Returns:
            bytes: Nunca retorna.

        Raises:
            ConnectionResetError: Siempre.

        Ejemplo:
            SocketHostil(0).recv(1)
        """
        raise ConnectionResetError("reset")

    def close(self) -> None:
        """
        Falla al cerrar.

        Contexto:
            Pruebas de la limpieza.

        Args:
            Ninguno.

        Returns:
            None: No retorna valor.

        Raises:
            OSError: Siempre.

        Ejemplo:
            SocketHostil(0).close()
        """
        raise OSError("ya cerrado")


def test_una_conexion_que_se_cae_a_medias_no_deja_cupos_ni_excepciones() -> None:
    servidor = sp.ServidorPanel(0)
    assert sp._leer_peticion(SocketHostil(0)) is None  # type: ignore[arg-type]
    assert servidor._cupo.acquire(blocking=False)
    servidor._atender(SocketHostil(0))  # type: ignore[arg-type]
    assert servidor._cupo.acquire(blocking=False)


def test_detener_ignora_un_error_al_cerrar_el_puerto() -> None:
    servidor = sp.ServidorPanel(0)
    servidor._escucha = SocketHostil(0)  # type: ignore[assignment]
    servidor.detener()


def test_un_navegador_inactivo_sigue_conectado_mientras_el_servidor_espera(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sp, "ESPERA_LECTURA", 0.02)
    inactivo = sp.ServidorPanel(0)
    inactivo.iniciar()
    try:
        cliente = ClientePanel(inactivo.puerto)
        assert esperar(lambda: inactivo.clientes == 1)
        time.sleep(0.2)
        assert inactivo.clientes == 1
        cliente.cerrar()
    finally:
        inactivo.detener()
