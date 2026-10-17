"""Pruebas de BridgeClient contra un juego falso que habla el protocolo por pipes reales."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, cast

import pytest

from dmbridge import Action, BridgeClient, BridgeError, BridgeTimeout
from dmbridge import client as client_module
from dmbridge.launcher import LaunchConfig
from tests.conftest import Session
from tests.support import CLOSE_PIPES, HANG, FakeProcess, ok_reply

MakeSession = Callable[..., Session]


def test_connect_envia_hello_con_la_version(make_session: MakeSession) -> None:
    """connect espera al juego y lo saluda con la version del protocolo."""
    session = make_session(lambda m: ok_reply(m, proto=1, game="falso"))
    info = session.client.connect(5.0)
    assert info["game"] == "falso"
    assert session.game.received == [{"op": "hello", "proto": 1, "v": 1}]
    assert session.client.last_latency > 0


def test_connect_sin_juego_vence_por_timeout(make_session: MakeSession) -> None:
    """Si el juego no abre los pipes connect levanta TimeoutError."""
    session = make_session(start_game=False)
    with pytest.raises(TimeoutError):
        session.client.connect(0.1)


def test_state_reset_config_y_exec_envian_la_orden_correcta(make_session: MakeSession) -> None:
    """Cada metodo de alto nivel envia su op con los campos esperados y devuelve la respuesta."""

    def handler(message: dict[str, Any]) -> dict[str, Any]:
        """
        Responde con un campo identificable por orden.

        Contexto:
            Manejador del juego falso en esta prueba.

        Args:
            message (dict[str, Any]): Orden recibida.

        Returns:
            dict[str, Any]: Respuesta ok.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        return ok_reply(message, result=42 if message["op"] == "exec" else None, t=1.5)

    session = make_session(handler)
    client = session.client
    client.connect(5.0)
    assert client.state()["t"] == 1.5
    assert client.reset()["op"] == "reset"
    assert client.config(dilation=0.5, console=["t.MaxFPS 0"], anim_tick_option=0)["op"] == "config"
    assert client.config()["op"] == "config"
    assert client.exec_lua("return 42") == 42
    received = session.game.received
    assert received[1] == {"op": "state", "v": 1}
    assert received[3] == {"op": "config", "dilation": 0.5, "console": ["t.MaxFPS 0"], "anim_tick_option": 0, "v": 1}
    assert received[4] == {"op": "config", "v": 1}
    assert received[5] == {"op": "exec", "lua": "return 42", "v": 1}


def test_step_serializa_la_accion_y_los_frames(make_session: MakeSession) -> None:
    """step envia n y la accion en su forma de cable."""
    session = make_session()
    session.client.connect(5.0)
    session.client.step(Action(btn="rb", lb=True, my=1.0), n_frames=3)
    assert session.game.received[-1] == {"op": "step", "n": 3, "a": {"btn": "rb", "lb": True, "my": 1.0}, "v": 1}


@pytest.mark.parametrize("frames", [0, -1, 601])
def test_step_valida_n_frames_antes_de_enviar(make_session: MakeSession, frames: int) -> None:
    """Un numero de frames fuera de [1, 600] se rechaza sin tocar el pipe."""
    session = make_session(start_game=False)
    with pytest.raises(ValueError, match="n_frames"):
        session.client.step(Action(), frames)


@pytest.mark.parametrize("frames", [1, 600])
def test_step_acepta_los_limites(make_session: MakeSession, frames: int) -> None:
    """Los extremos 1 y 600 son validos."""
    session = make_session()
    session.client.connect(5.0)
    session.client.step(Action(), frames)
    assert session.game.received[-1]["n"] == frames


def test_respuesta_ok_false_levanta_bridge_error(make_session: MakeSession) -> None:
    """Una respuesta con ok=false se convierte en BridgeError con op y motivo."""
    session = make_session(lambda m: {"op": m["op"], "ok": False, "err": "no admitido", "v": 1})
    with pytest.raises(BridgeError, match="hello: no admitido"):
        session.client.connect(5.0)


def test_respuesta_sin_ok_se_trata_como_fallo(make_session: MakeSession) -> None:
    """Una respuesta sin campo ok no es exitosa."""
    session = make_session(lambda m: {"op": m["op"], "v": 1})
    with pytest.raises(BridgeError):
        session.client.connect(5.0)


def test_respuesta_con_version_distinta_levanta_value_error(make_session: MakeSession) -> None:
    """Una respuesta de otra version del protocolo no se acepta."""
    session = make_session(lambda m: {"op": m["op"], "ok": True, "v": 2})
    with pytest.raises(ValueError, match="version"):
        session.client.connect(5.0)


def test_pipe_cerrado_por_el_juego_levanta_bridge_error(make_session: MakeSession) -> None:
    """Si el juego cierra sin responder la orden falla con BridgeError (no BridgeTimeout)."""
    session = make_session(lambda _m: CLOSE_PIPES)
    with pytest.raises(BridgeError, match="conexion perdida") as info:
        session.client.connect(5.0)
    assert not isinstance(info.value, BridgeTimeout)


def test_request_sin_conectar_levanta_bridge_error(make_session: MakeSession) -> None:
    """Enviar antes de conectar es una conexion perdida, no un cuelgue."""
    session = make_session(start_game=False)
    with pytest.raises(BridgeError, match="conexion perdida"):
        session.client.state()


def test_vigilante_mata_la_instancia_y_levanta_bridge_timeout(make_session: MakeSession) -> None:
    """Si una respuesta excede el plazo el vigilante mata el proceso y request levanta BridgeTimeout."""

    def handler(message: dict[str, Any]) -> Any:
        """
        Responde al saludo y deja colgada cualquier otra orden.

        Contexto:
            Simula un juego congelado tras conectar.

        Args:
            message (dict[str, Any]): Orden recibida.

        Returns:
            Any: Respuesta ok para hello, HANG para el resto.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        return ok_reply(message) if message["op"] == "hello" else HANG

    session = make_session(handler)
    process = FakeProcess(on_kill=session.game._stop.set)
    session.client.process = process  # type: ignore[assignment]
    session.client.connect(5.0)
    started = time.monotonic()
    with pytest.raises(BridgeTimeout, match="instancia terminada"):
        session.client.request({"op": "state"}, timeout=0.3)
    assert process.killed >= 1
    assert time.monotonic() - started < 3.0


def test_vigilante_no_actua_sin_proceso_ni_con_proceso_terminado(make_session: MakeSession) -> None:
    """Sin proceso, o con uno que ya salio, vencer el plazo no intenta matar nada."""
    session = make_session()
    session.client.connect(5.0)
    with session.client._lock:
        session.client._deadline = time.monotonic() - 1.0
    time.sleep(0.6)
    assert not session.client._timed_out
    process = FakeProcess()
    process.killed = 1
    session.client.process = process  # type: ignore[assignment]
    time.sleep(0.6)
    assert process.killed == 1
    with session.client._lock:
        session.client._deadline = None


def test_el_plazo_se_limpia_tras_cada_request(make_session: MakeSession) -> None:
    """Terminada una orden el vigilante no conserva un plazo viejo."""
    session = make_session()
    session.client.connect(5.0)
    assert session.client._deadline is None


def test_close_con_quit_game_envia_close_y_espera_al_proceso(make_session: MakeSession) -> None:
    """close(quit_game=True) pide cerrar al juego y espera la salida del proceso."""
    session = make_session()
    process = FakeProcess()
    session.client.process = process  # type: ignore[assignment]
    session.client.connect(5.0)
    session.client.close()
    assert session.game.received[-1]["op"] == "close"
    assert process.wait_calls == 1
    assert process.killed == 0
    assert session.client._stop.is_set()


def test_close_sin_quit_game_no_envia_nada(make_session: MakeSession) -> None:
    """close(quit_game=False) solo libera los recursos locales."""
    session = make_session()
    session.client.connect(5.0)
    session.client.close(quit_game=False)
    assert [m["op"] for m in session.game.received] == ["hello"]


def test_close_mata_el_proceso_si_no_sale_a_tiempo(make_session: MakeSession) -> None:
    """Si el juego no sale tras close el proceso se termina y se espera de nuevo."""
    session = make_session()
    process = FakeProcess(wait_timeouts=1)
    session.client.process = process  # type: ignore[assignment]
    session.client.connect(5.0)
    session.client.close()
    assert process.killed == 1
    assert process.wait_calls == 2


def test_close_ignora_errores_del_juego(make_session: MakeSession) -> None:
    """Si el juego ya no responde (BridgeError) o responde con otra version (ValueError), close termina igual."""
    session = make_session(lambda m: {"op": m["op"], "ok": True, "v": 9} if m["op"] == "close" else ok_reply(m))
    session.client.connect(5.0)
    session.client.close()
    sin_conexion = make_session(start_game=False)
    sin_conexion.client.close()


def test_launch_aplica_la_dilatacion_de_la_configuracion(
    make_session: MakeSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """launch fuerza el instance_id del cliente, lanza, saluda y aplica dilation()=0.5 con sim_hz=60."""
    session = make_session()
    process = FakeProcess()
    launched: list[LaunchConfig] = []

    def fake_launch(config: LaunchConfig) -> FakeProcess:
        """
        Sustituye a launch_instance sin lanzar ningun proceso.

        Contexto:
            Parche de dmbridge.client.launch_instance.

        Args:
            config (LaunchConfig): Configuracion recibida.

        Returns:
            FakeProcess: Proceso falso.

        Raises:
            Ninguna.

        Ejemplo:
            (uso interno)
        """
        launched.append(config)
        return process

    monkeypatch.setattr(client_module, "launch_instance", fake_launch)
    info = session.client.launch(LaunchConfig(instance_id="otro", sim_hz=60.0), connect_timeout=5.0)
    assert info["op"] == "hello"
    assert launched[0].instance_id == session.client.instance_id
    assert cast(object, session.client.process) is process
    assert [m["op"] for m in session.game.received] == ["hello", "config"]
    assert session.game.received[1]["dilation"] == 0.5


def test_launch_sin_dilatacion_a_30hz_y_configuracion_por_defecto(
    make_session: MakeSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Con la configuracion por defecto (30 Hz) no se envia config tras el saludo."""
    session = make_session()
    monkeypatch.setattr(client_module, "launch_instance", lambda _c: FakeProcess())
    session.client.launch(connect_timeout=5.0)
    assert [m["op"] for m in session.game.received] == ["hello"]


def test_launch_mata_el_proceso_si_no_se_conecta(make_session: MakeSession, monkeypatch: pytest.MonkeyPatch) -> None:
    """Si el juego no se conecta a tiempo launch termina la instancia y propaga TimeoutError."""
    session = make_session(start_game=False)
    process = FakeProcess()
    monkeypatch.setattr(client_module, "launch_instance", lambda _c: process)
    with pytest.raises(TimeoutError):
        session.client.launch(connect_timeout=0.1)
    assert process.killed == 1


def test_el_cliente_expone_el_identificador_y_el_plazo(instance_id: str) -> None:
    """El constructor conserva instance_id y reply_timeout, y arranca un vigilante demonio."""
    client = BridgeClient(instance_id, reply_timeout=1.5)
    try:
        assert (client.instance_id, client.reply_timeout) == (instance_id, 1.5)
        assert client._watchdog.daemon
        assert client._watchdog.is_alive()
    finally:
        client.close(quit_game=False)
    client._watchdog.join(timeout=3.0)
    assert not client._watchdog.is_alive()


def test_reset_envia_los_parametros_de_curriculo(make_session: MakeSession) -> None:
    """reset con multiplicador y fase de inicio los incluye en la orden; sin ellos no los envia."""
    session = make_session()
    client = session.client
    client.connect(5.0)
    client.reset()
    client.reset(boss_hp_mult=0.30000001, start_phase=2)
    assert session.game.received[1] == {"op": "reset", "v": 1}
    assert session.game.received[2] == {"op": "reset", "boss_hp_mult": 0.3, "start_phase": 2, "v": 1}


def test_reset_con_descripcion_de_jefe_envia_boss_y_la_recarga_solo_si_se_pide(make_session: MakeSession) -> None:
    """boss viaja tal cual; reload solo se envia con boss y solo cuando es verdadero."""
    session = make_session()
    client = session.client
    client.connect(5.0)
    client.reset(boss={"id": "amlo"})
    client.reset(boss={"id": "amlo"}, reload=True)
    client.reset(reload=True)
    assert session.game.received[1] == {"op": "reset", "boss": {"id": "amlo"}, "v": 1}
    assert session.game.received[2] == {"op": "reset", "boss": {"id": "amlo"}, "reload": True, "v": 1}
    assert session.game.received[3] == {"op": "reset", "v": 1}


@pytest.mark.parametrize("kwargs", [{"boss_hp_mult": 0.0}, {"boss_hp_mult": 1.5}, {"start_phase": 3}])
def test_reset_rechaza_parametros_de_curriculo_invalidos(make_session: MakeSession, kwargs: dict[str, Any]) -> None:
    """Los valores fuera de la lista de admitidos se rechazan antes de enviar nada al juego."""
    session = make_session()
    session.client.connect(5.0)
    with pytest.raises(ValueError):
        session.client.reset(**kwargs)
    assert len(session.game.received) == 1
