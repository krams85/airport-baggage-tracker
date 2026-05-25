"""
Real-time web dashboard for Airport Baggage Tracker.
Opens at http://localhost:8765  (configurable port).
Requires: pip install aiohttp
"""
import asyncio
import base64
import json
import logging
import threading
from datetime import datetime
from typing import Optional, Set

import cv2
import numpy as np

logger = logging.getLogger("BaggageTracker.Web")

_AIOHTTP_OK = False
try:
    from aiohttp import web as _aio_web
    _AIOHTTP_OK = True
except ImportError:
    pass

DEFAULT_PORT = 8765

# ── Embedded single-page dashboard ────────────────────────────────────────────

_HTML = r"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Baggage Tracker — Live</title>
<style>
:root{
  --base:#1e1e2e;--mantle:#181825;--crust:#11111b;
  --s0:#313244;--s1:#45475a;--s2:#585b70;
  --ov0:#6c7086;--text:#cdd6f4;--sub:#a6adc8;
  --blue:#89b4fa;--green:#a6e3a1;--yellow:#f9e2af;--red:#f38ba8;
  --mauve:#cba6f7;--teal:#94e2d5;
}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:"Segoe UI",system-ui,sans-serif;background:var(--base);color:var(--text);min-height:100vh}

/* ── header ── */
header{
  background:var(--mantle);border-bottom:1px solid var(--s0);
  padding:14px 24px;display:flex;align-items:center;gap:12px;
  position:sticky;top:0;z-index:10;
  backdrop-filter:blur(8px);
}
header h1{font-size:20px;font-weight:700;color:var(--blue);letter-spacing:-.3px}
.sub{color:var(--ov0);font-size:12px;margin-top:2px}
.ws-row{margin-left:auto;display:flex;align-items:center;gap:8px;font-size:12px;color:var(--sub);white-space:nowrap}
.dot{width:9px;height:9px;border-radius:50%;background:var(--ov0);flex-shrink:0;transition:.3s}
.dot.ok{background:var(--green);box-shadow:0 0 7px var(--green)}
.dot.err{background:var(--red)}

/* ── stats bar ── */
.stats{display:flex;gap:10px;padding:14px 24px;background:var(--mantle);border-bottom:1px solid var(--s0)}
.sc{flex:1;background:var(--crust);border:1px solid var(--s0);border-radius:10px;
    padding:14px 10px;text-align:center;min-width:0}
.sc .v{font-size:30px;font-weight:800;line-height:1;font-variant-numeric:tabular-nums}
.sc .l{font-size:10px;color:var(--ov0);margin-top:4px;text-transform:uppercase;letter-spacing:.6px}
.sc.total .v{color:var(--blue)}.sc.same .v{color:var(--green)}
.sc.prob .v{color:var(--yellow)}.sc.diff .v{color:var(--red)}

/* ── toolbar ── */
.toolbar{display:flex;gap:6px;padding:12px 24px 6px;align-items:center;flex-wrap:wrap;
         border-bottom:1px solid var(--s0)}
.toolbar .lbl{font-size:12px;color:var(--ov0);margin-right:2px}
.fb{padding:4px 14px;border-radius:20px;border:1px solid var(--s1);background:var(--s0);
    color:var(--sub);cursor:pointer;font-size:12px;transition:.2s;line-height:1.6}
.fb:hover{border-color:var(--blue);color:var(--text)}
.fb.a-all  {background:var(--blue);  border-color:var(--blue);  color:var(--crust);font-weight:600}
.fb.a-same {background:var(--green); border-color:var(--green); color:var(--crust);font-weight:600}
.fb.a-prob {background:var(--yellow);border-color:var(--yellow);color:var(--crust);font-weight:600}
.fb.a-diff {background:var(--red);   border-color:var(--red);   color:var(--crust);font-weight:600}
.ml{margin-left:auto}
.clr{padding:4px 14px;border-radius:6px;border:1px solid var(--s1);
     background:transparent;color:var(--ov0);cursor:pointer;font-size:12px;transition:.2s}
.clr:hover{background:var(--s0);color:var(--text)}

/* ── feed ── */
.feed{padding:10px 24px 40px;display:flex;flex-direction:column;gap:10px}

/* ── match card ── */
.card{
  background:var(--mantle);border-radius:12px;border:2px solid var(--s1);
  padding:16px;display:flex;gap:16px;align-items:center;
  animation:si .35s ease;transition:border-color .3s;
}
@keyframes si{from{opacity:0;transform:translateY(-12px)}to{opacity:1;transform:none}}
.card.same{border-color:var(--green)}.card.prob{border-color:var(--yellow)}.card.diff{border-color:var(--red)}

.cam{display:flex;flex-direction:column;align-items:center;gap:8px;min-width:110px}
.cam img{
  width:110px;height:110px;object-fit:cover;border-radius:8px;
  border:1px solid var(--s0);background:var(--crust);display:block;
}
.cam .no-img{width:110px;height:110px;background:var(--crust);border-radius:8px;
             display:flex;align-items:center;justify-content:center;
             font-size:28px;color:var(--s2)}
.cl{font-size:11px;color:var(--sub);text-align:center;line-height:1.5}
.cn{color:var(--text);font-weight:600;font-size:12px}

.vb{flex:1;display:flex;flex-direction:column;align-items:center;gap:6px}
.arr{font-size:28px;color:var(--s2)}
.pct{font-size:40px;font-weight:800;line-height:1;font-variant-numeric:tabular-nums}
.vt{font-size:13px;font-weight:700}
.ts{font-size:11px;color:var(--ov0);margin-top:2px}
.same .pct,.same .vt{color:var(--green)}
.prob .pct,.prob .vt{color:var(--yellow)}
.diff .pct,.diff .vt{color:var(--red)}

/* ── empty state ── */
.empty{text-align:center;padding:80px 24px;color:var(--ov0);display:flex;
       flex-direction:column;align-items:center;gap:12px}
.empty .ico{font-size:56px}
.empty p{font-size:16px}
.empty small{font-size:13px;color:var(--s2)}

/* ── scrollbar ── */
::-webkit-scrollbar{width:5px}
::-webkit-scrollbar-thumb{background:var(--s1);border-radius:3px}

@media(max-width:600px){
  .stats{flex-wrap:wrap}.cam img,.cam .no-img{width:80px;height:80px}
  .cam{min-width:90px}.pct{font-size:28px}.vb{gap:4px}
  header h1{font-size:16px}.sub{display:none}
}
</style>
</head>
<body>
<header>
  <div>
    <h1>✈ Baggage Tracker</h1>
    <div class="sub">Мониторинг совпадений · реальное время</div>
  </div>
  <div class="ws-row">
    <div class="dot" id="d"></div>
    <span id="wst">Подключение…</span>
  </div>
</header>

<div class="stats">
  <div class="sc total"><div class="v" id="sT">0</div><div class="l">Всего событий</div></div>
  <div class="sc same"> <div class="v" id="sS">0</div><div class="l">✔ Тот же</div></div>
  <div class="sc prob"> <div class="v" id="sP">0</div><div class="l">? Вероятно</div></div>
  <div class="sc diff"> <div class="v" id="sD">0</div><div class="l">✘ Другой</div></div>
</div>

<div class="toolbar">
  <span class="lbl">Фильтр:</span>
  <button class="fb a-all"  id="fb-all"  onclick="filt('all')">Все</button>
  <button class="fb"        id="fb-same" onclick="filt('same')">✔ Тот же</button>
  <button class="fb"        id="fb-prob" onclick="filt('prob')">? Вероятно</button>
  <button class="fb"        id="fb-diff" onclick="filt('diff')">✘ Другой</button>
  <button class="clr ml"                 onclick="clrFeed()">🗑 Очистить</button>
</div>

<div class="feed" id="feed">
  <div class="empty" id="em">
    <div class="ico">🧳</div>
    <p>Ожидание событий совпадения…</p>
    <small>Запустите приложение в Рабочем режиме</small>
  </div>
</div>

<script>
var all=[], f='all', st={T:0,S:0,P:0,D:0};

function vc(v){ return v.indexOf('Тот же')>=0?'same':v.indexOf('Вероятно')>=0?'prob':'diff'; }

function conn(){
  var ws=new WebSocket('ws://'+location.host+'/ws');
  ws.onopen=function(){ document.getElementById('d').className='dot ok'; document.getElementById('wst').textContent='Подключено'; };
  ws.onclose=function(){ document.getElementById('d').className='dot err'; document.getElementById('wst').textContent='Переподключение…'; setTimeout(conn,3000); };
  ws.onerror=function(){ document.getElementById('d').className='dot err'; };
  ws.onmessage=function(e){
    var d=JSON.parse(e.data);
    if(d.type==='match') addMatch(d);
    else if(d.type==='stats') applyStats(d);
  };
}

function addMatch(d){
  var c=vc(d.verdict);
  all.unshift({d:d,c:c});
  if(all.length>200) all.pop();
  st.T++; if(c==='same')st.S++; else if(c==='prob')st.P++; else st.D++;
  updSt(); render();
}

function applyStats(d){
  st={T:d.total,S:d.same,P:d.probable,D:d.different};
  updSt();
}

function updSt(){
  document.getElementById('sT').textContent=st.T;
  document.getElementById('sS').textContent=st.S;
  document.getElementById('sP').textContent=st.P;
  document.getElementById('sD').textContent=st.D;
}

function render(){
  var feed=document.getElementById('feed');
  var cards=feed.querySelectorAll('.card');
  for(var i=0;i<cards.length;i++) cards[i].parentNode.removeChild(cards[i]);
  var vis=(f==='all')?all:all.filter(function(x){return x.c===f;});
  document.getElementById('em').style.display=vis.length?'none':'';
  var frag=document.createDocumentFragment();
  vis.slice(0,100).forEach(function(x){ frag.appendChild(mkCard(x.d,x.c)); });
  feed.appendChild(frag);
}

function mkCard(d,c){
  var el=document.createElement('div');
  el.className='card '+c;
  var si=d.source_img?('<img src="'+d.source_img+'" loading="lazy">'):'<div class="no-img">📦</div>';
  var qi=d.query_img? ('<img src="'+d.query_img+'"  loading="lazy">'):'<div class="no-img">📦</div>';
  el.innerHTML=
    '<div class="cam">'+si+'<div class="cl"><div class="cn">📤 '+h(d.source_cam)+'</div>Стол #'+d.source_desk+' · Трек #'+d.source_track+'</div></div>'+
    '<div class="vb '+c+'"><div class="arr">→</div><div class="pct">'+d.similarity+'%</div>'+
    '<div class="vt">'+h(d.verdict)+'</div><div class="ts">'+d.timestamp+'</div></div>'+
    '<div class="cam">'+qi+'<div class="cl"><div class="cn">📥 '+h(d.query_cam)+'</div>Стол #'+d.query_desk+' · Трек #'+d.query_track+'</div></div>';
  return el;
}

var FB_MAP={all:'a-all',same:'a-same',prob:'a-prob',diff:'a-diff'};
function filt(v){
  f=v;
  ['all','same','prob','diff'].forEach(function(k){
    var b=document.getElementById('fb-'+k);
    b.className='fb'+(k===v?' '+FB_MAP[k]:'');
  });
  render();
}

function clrFeed(){
  all=[]; st={T:0,S:0,P:0,D:0}; updSt(); render();
}

function h(s){ return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

conn();
</script>
</body>
</html>"""


# ── WebDashboard class ─────────────────────────────────────────────────────────

class WebDashboard:
    """
    Asyncio-based web + WebSocket server running in a daemon thread.
    Call start() once, then push_match(mr) from any thread.
    """

    def __init__(self, port: int = DEFAULT_PORT):
        self._port    = port
        self._loop:   Optional[asyncio.AbstractEventLoop] = None
        self._clients: Set = set()
        self._history: list = []          # last 100 encoded match dicts
        self._stats = dict(total=0, same=0, probable=0, different=0)
        self._running = False

    @property
    def port(self) -> int:
        return self._port

    @property
    def url(self) -> str:
        return f"http://localhost:{self._port}"

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> bool:
        if not _AIOHTTP_OK:
            logger.warning(
                "aiohttp не установлен — веб-дашборд отключён. "
                "Установите: pip install aiohttp"
            )
            return False
        if self._running:
            return True
        self._loop = asyncio.new_event_loop()
        t = threading.Thread(target=self._thread_main, daemon=True, name="WebDashboard")
        t.start()
        self._running = True
        logger.info("Веб-дашборд запускается на %s", self.url)
        return True

    def stop(self):
        if self._loop and self._running:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._running = False

    def push_match(self, mr) -> None:
        """Thread-safe. Call from GUI thread after a MatchResult arrives."""
        if not self._running or self._loop is None:
            return
        data = self._encode_match(mr)
        self._history.append(data)
        if len(self._history) > 100:
            self._history.pop(0)
        v = mr.verdict
        self._stats["total"] += 1
        if "Тот же" in v:
            self._stats["same"] += 1
        elif "Вероятно" in v:
            self._stats["probable"] += 1
        else:
            self._stats["different"] += 1
        asyncio.run_coroutine_threadsafe(self._broadcast(data), self._loop)

    def reset(self):
        self._stats = dict(total=0, same=0, probable=0, different=0)
        self._history.clear()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _thread_main(self):
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        except Exception as exc:
            logger.error("Web server error: %s", exc)

    async def _serve(self):
        app = _aio_web.Application()
        app.router.add_get("/",         self._h_index)
        app.router.add_get("/ws",       self._h_ws)
        app.router.add_get("/api/stats", self._h_stats)
        runner = _aio_web.AppRunner(app)
        await runner.setup()
        site = _aio_web.TCPSite(runner, "localhost", self._port)
        await site.start()
        await asyncio.Event().wait()    # run until loop.stop()

    async def _h_index(self, _req):
        return _aio_web.Response(text=_HTML, content_type="text/html", charset="utf-8")

    async def _h_stats(self, _req):
        return _aio_web.json_response({**self._stats, "type": "stats"})

    async def _h_ws(self, request):
        ws = _aio_web.WebSocketResponse(heartbeat=30)
        await ws.prepare(request)
        self._clients.add(ws)
        # Replay last 20 events on connect
        for d in self._history[-20:]:
            await ws.send_str(json.dumps(d))
        await ws.send_str(json.dumps({**self._stats, "type": "stats"}))
        try:
            async for _ in ws:
                pass
        finally:
            self._clients.discard(ws)
        return ws

    async def _broadcast(self, data: dict):
        msg = json.dumps(data)
        dead = set()
        for ws in set(self._clients):
            try:
                await ws.send_str(msg)
            except Exception:
                dead.add(ws)
        self._clients -= dead

    # ── Encoding helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _encode_match(mr) -> dict:
        return {
            "type":         "match",
            "timestamp":    datetime.fromtimestamp(mr.timestamp).strftime("%H:%M:%S"),
            "verdict":      mr.verdict,
            "verdict_color": mr.verdict_color,
            "similarity":   round(mr.similarity * 100, 1),
            "source_cam":   mr.source_entry.cam_name,
            "source_desk":  mr.source_entry.counter_id,
            "source_track": mr.source_entry.track_id,
            "query_cam":    mr.query_cam_name,
            "query_desk":   mr.query_counter_id,
            "query_track":  mr.query_track_id,
            "source_img":   WebDashboard._b64img(mr.source_entry.crop),
            "query_img":    WebDashboard._b64img(mr.query_crop),
        }

    @staticmethod
    def _b64img(crop_bgr: Optional[np.ndarray]) -> str:
        if crop_bgr is None or crop_bgr.size == 0:
            return ""
        try:
            ok, buf = cv2.imencode(
                ".jpg", crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 75]
            )
            return ("data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
                    if ok else "")
        except Exception:
            return ""
