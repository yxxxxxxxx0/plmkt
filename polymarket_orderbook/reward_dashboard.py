"""Live dashboard for a reward_quoter.py run: rewards against fill P&L, quotes against the book, fills, the log.

    python reward_dashboard.py                                   data/rewards_paper, logs/reward_paper.log
    python reward_dashboard.py --out data/rewards --log logs/reward_quoter.log --port 8766

then open http://127.0.0.1:8765. It reads the run's files on every refresh (30 s); it never writes them.
"""
import argparse
import csv
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_POINTS = 1500
RECENT_S = 2 * 3600     # the window the reward pace is measured over


def rows(path):
    if not os.path.exists(path) or not os.path.getsize(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def thin(xs):
    k = max(1, len(xs) // MAX_POINTS)
    return xs[::k] + ([xs[-1]] if xs and (len(xs) - 1) % k else [])


def data(out, log):
    st = json.load(open(os.path.join(out, "state.json")))
    M, F = rows(os.path.join(out, "minutes.csv")), rows(os.path.join(out, "fills.csv"))
    names = {**st.get("seen", {}), **{m["cond"]: m["q"] for m in st["markets"]}}
    current = [m["cond"] for m in st["markets"]]
    bank = st["bankroll"]

    # one point per sample instant: rewards so far, fills marked at the mid, and their sum
    by_ts, cum, timeline = {}, 0.0, []
    for r in M:
        by_ts.setdefault(r["ts"], []).append(r)
    for ts in sorted(by_ts, key=float):
        rs = by_ts[ts]
        cum += sum(num(r["reward_min"], 0) for r in rs)
        pos = 0.0
        for r in rs:
            n, mid = num(r["net"], 0), num(r["mid"])
            if mid is not None:
                pos += n * mid if n > 0 else -n * (1 - mid)
        fill = num(rs[0]["cash"], bank) + pos - bank
        timeline.append([float(ts), round(cum, 4), round(fill, 4), round(cum + fill, 4)])

    # per market: totals, and its quotes against the book over time
    mk = {}
    for r in M:
        c = r["cond"]
        g = mk.setdefault(c, dict(cond=c, q=names.get(c, c[:12]), mins=0, quoted=0, reward=0.0, share=0.0, px=[],
                                  current=c in current))
        g["mins"] += 1
        g["quoted"] += num(r["my_q"], 0) > 0
        g["reward"] += num(r["reward_min"], 0)
        g["share"] += num(r["share"], 0)
        g["px"].append([float(r["ts"]), num(r["mid"]), num(r["best_bid"]), num(r["best_ask"]), num(r["my_bid"]),
                        num(r["my_ask"])])
        g["last"] = r
    for c, g in mk.items():
        r = g.pop("last")
        g.update(share=g["share"] / g["mins"], quoted=g["quoted"] / g["mins"], net=num(r["net"], 0), mid=num(r["mid"]),
                 bid=num(r["best_bid"]), ask=num(r["best_ask"]), my_bid=num(r["my_bid"]), my_ask=num(r["my_ask"]),
                 fills=[[float(f["ts"]), float(f["price"]), f["side"], float(f["qty"]), f["how"]] for f in F if f["cond"] == c])
        g["px"] = thin(g["px"])
    markets = sorted(mk.values(), key=lambda g: (not g["current"], -g["reward"]))

    last_fill = timeline[-1][2] if timeline else 0.0
    # the pace now, not the average since the start: a market's first hour usually pays far more than the rest
    recent = [r for r in timeline if r[0] >= timeline[-1][0] - RECENT_S] if timeline else []
    span = (recent[-1][0] - recent[0][0]) if len(recent) > 1 else 0.0
    pace = (recent[-1][1] - recent[0][1]) / span * 86400 if span >= 1800 else None
    days = (timeline[-1][0] - timeline[0][0]) / 86400 if len(timeline) > 1 else 0.0
    tail = []
    if log and os.path.exists(log):
        with open(log, encoding="utf-8", errors="replace") as f:
            tail = [l.rstrip() for l in f.readlines()[-40:] if l.strip()]
    paused = {names.get(next((m["cond"] for m in st["markets"] if m["yes"] == t), ""), t[:10]): p[1]
              for t, p in st.get("paused", {}).items()}
    return dict(
        updated=max([float(timeline[-1][0])] if timeline else [0]),
        summary=dict(reward=st["reward"], fill=last_fill, pace=pace, pace_h=span / 3600, net=st["reward"] + last_fill, fills=len(F), days=days,
                     bankroll=bank, cash=st["cash"], quoted=sum(g["quoted"] * g["mins"] for g in mk.values()) /
                     max(1, sum(g["mins"] for g in mk.values())), paused=paused,
                     held={names.get(g["cond"]): g["net"] for g in markets if abs(g["net"]) >= 1}),
        timeline=thin(timeline), markets=markets,
        fills=[dict(ts=float(f["ts"]), q=names.get(f["cond"], f["cond"][:12]), side=f["side"], price=float(f["price"]),
                    qty=float(f["qty"]), how=f["how"], net=float(f["net_after"])) for f in F[-60:]][::-1],
        log=tail)


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Reward Quoter Monitor</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root{color-scheme:light;--surface-0:#f4f4f2;--surface-1:#fcfcfb;--border:#e2e1dc;--text-primary:#0b0b0b;
 --text-secondary:#52514e;--text-muted:#7b7a75;--grid:#ebeae6;--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--mid:#7b7a75;
 --good:#008300;--bad:#e34948;--warn:#eda100}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--surface-0:#121211;--surface-1:#1a1a19;
 --border:#2e2e2c;--text-primary:#f0efec;--text-secondary:#c3c2b7;--text-muted:#8f8e88;--grid:#262624;--s1:#3987e5;
 --s2:#d95926;--s3:#199e70;--mid:#8f8e88;--good:#3fa83f;--bad:#e66767;--warn:#c98500}}
:root[data-theme="dark"]{color-scheme:dark;--surface-0:#121211;--surface-1:#1a1a19;--border:#2e2e2c;--text-primary:#f0efec;
 --text-secondary:#c3c2b7;--text-muted:#8f8e88;--grid:#262624;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--mid:#8f8e88;
 --good:#3fa83f;--bad:#e66767;--warn:#c98500}
*{box-sizing:border-box}
body{margin:0;background:var(--surface-0);color:var(--text-primary);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1200px;margin:0 auto;padding:20px 16px 40px}
header{display:flex;justify-content:space-between;align-items:baseline;gap:12px;flex-wrap:wrap;margin-bottom:16px}
h1{font-size:20px;margin:0}h2{font-size:15px;margin:0 0 4px}
.sub{color:var(--text-secondary);font-size:13px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:16px}
.tile,.card{background:var(--surface-1);border:1px solid var(--border);border-radius:10px;padding:14px 16px}
.tile .k{color:var(--text-secondary);font-size:12px}.tile .v{font-size:24px;font-weight:600;font-variant-numeric:tabular-nums}
.tile .n{color:var(--text-muted);font-size:12px}
.pos{color:var(--good)}.neg{color:var(--bad)}
.card{margin-bottom:16px}.chart{position:relative;height:280px}
.legend{display:flex;gap:16px;flex-wrap:wrap;margin:4px 0 10px;color:var(--text-secondary);font-size:12px}
.legend i{display:inline-block;width:12px;height:3px;border-radius:2px;margin-right:6px;vertical-align:middle}
.legend i.dot{width:9px;height:9px;border-radius:50%}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:13px}
th,td{text-align:right;padding:6px 8px;border-bottom:1px solid var(--border);white-space:nowrap}
th{color:var(--text-secondary);font-weight:500}th:first-child,td:first-child{text-align:left;white-space:normal}
.scroll{overflow-x:auto}
tr.sel{background:color-mix(in srgb,var(--s1) 10%,transparent)}tbody tr.click{cursor:pointer}
.tag{font-size:11px;padding:1px 6px;border-radius:9px;border:1px solid var(--border);color:var(--text-secondary)}
pre{margin:0;max-height:260px;overflow:auto;font:12px/1.5 ui-monospace,Consolas,monospace;color:var(--text-secondary);white-space:pre-wrap}
.note{color:var(--text-muted);font-size:12px;margin-top:6px}
.two{display:grid;grid-template-columns:1fr 1fr;gap:16px}@media(max-width:800px){.two{grid-template-columns:1fr}}
.two .card{margin-bottom:0;min-width:0}
.tiles>*,.card{min-width:0}
.warn{border-left:3px solid var(--warn)}
</style></head><body><main>
<header><div><h1>Reward quoter</h1><div class="sub" id="sub">loading…</div></div>
<div class="sub" id="upd"></div></header>
<div class="tiles" id="tiles"></div>
<div id="alerts"></div>
<div class="card"><h2>Rewards against fill P&amp;L</h2>
<div class="sub">Cumulative dollars. Fill P&amp;L marks any position at the mid; net is the two added.</div>
<div class="legend"><span><i style="background:var(--s1)"></i>Rewards (estimated)</span>
<span><i style="background:var(--s2)"></i>Fill P&amp;L</span><span><i style="background:var(--s3)"></i>Net</span></div>
<div class="chart"><canvas id="pnl"></canvas></div></div>
<div class="card"><h2 id="pxh">Quotes against the book</h2>
<div class="sub">YES price. Click a market in the table below to switch.</div>
<div class="legend"><span><i style="background:var(--mid)"></i>Market mid</span>
<span><i style="background:var(--s1)"></i>My bid</span><span><i style="background:var(--s2)"></i>My ask</span>
<span><i class="dot" style="background:var(--s3)"></i>Bought</span><span><i class="dot" style="background:var(--bad)"></i>Sold</span></div>
<div class="chart"><canvas id="px"></canvas></div></div>
<div class="card"><h2>Markets</h2><div class="scroll"><table id="mk"></table></div>
<div class="note">Share: my reward points as a fraction of everyone's in the band, averaged over the minutes logged.</div></div>
<div class="two"><div class="card"><h2>Fills</h2><div class="scroll" style="max-height:300px"><table id="fl"></table></div></div>
<div class="card"><h2>Log</h2><pre id="log"></pre></div></div>
</main>
<script>
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const usd=(x,s)=>(s&&x>0?'+':'')+(x<0?'−$':'$')+Math.abs(x).toFixed(2);
const tfmt=t=>new Date(t*1000).toLocaleString([], {month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'});
let D=null, sel=null, charts={};
function axis(){return{ticks:{color:css('--text-muted'),font:{size:11}},grid:{color:css('--grid')},border:{display:false}}}
function xaxis(){const a=axis();a.type='linear';a.ticks.callback=v=>new Date(v*1000).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'});a.ticks.maxTicksLimit=8;a.grid.display=false;return a}
function base(ycb){return{animation:false,maintainAspectRatio:false,interaction:{mode:'nearest',axis:'x',intersect:false},
 plugins:{legend:{display:false},tooltip:{backgroundColor:css('--surface-1'),titleColor:css('--text-primary'),bodyColor:css('--text-secondary'),
 borderColor:css('--border'),borderWidth:1,padding:10,callbacks:{title:i=>i.length?tfmt(i[0].parsed.x):''}}},
 scales:{x:xaxis(),y:Object.assign(axis(),{ticks:Object.assign(axis().ticks,{callback:ycb})})}}}
function line(label,data,color,extra){return Object.assign({label,data,borderColor:color,backgroundColor:color,borderWidth:2,pointRadius:0,pointHoverRadius:4,tension:0},extra||{})}
function draw(id,cfg){if(charts[id])charts[id].destroy();charts[id]=new Chart(document.getElementById(id),cfg)}
function tiles(){const s=D.summary;const t=[
 ['Net',usd(s.net,1),'rewards + fill P&L',s.net>0.005?'pos':s.net<-0.005?'neg':''],
 ['Rewards',usd(s.reward),'estimated, not yet paid',''],
 ['Fill P&L',usd(s.fill,1),'marked at the mid',s.fill>0.005?'pos':s.fill<-0.005?'neg':''],
 ['Fills',s.fills,'',''],
 ['Quoted',(100*s.quoted).toFixed(0)+'%','of minutes logged',''],
 ['Rewards per day',s.pace==null?'—':usd(s.pace),s.pace==null?'needs 30 min of samples':'pace over the last '+s.pace_h.toFixed(1)+' h',''],
 ['Running',s.days.toFixed(2)+' days','on a $'+s.bankroll.toFixed(0)+' bankroll','']];
 document.getElementById('tiles').innerHTML=t.map(([k,v,n,c])=>`<div class="tile"><div class="k">${k}</div><div class="v ${c}">${v}</div><div class="n">${n}</div></div>`).join('');
 const al=[];const held=Object.entries(s.held);
 if(held.length)al.push('Holding: '+held.map(([q,n])=>`${q.slice(0,50)} ${n>0?'+':''}${n.toFixed(0)} YES`).join('; '));
 for(const [q,w] of Object.entries(s.paused))al.push(`Paused: ${q.slice(0,50)} (${w})`);
 if(s.pace!=null&&s.pace<1)al.push(`Rewards are running at ${usd(s.pace)}/day, under Polymarket's $1/day payout minimum: at this pace nothing would be paid.`);
 document.getElementById('alerts').innerHTML=al.map(a=>`<div class="card warn sub">${a}</div>`).join('');}
function pnl(){const T=D.timeline;draw('pnl',{type:'line',data:{datasets:[
 line('Rewards',T.map(r=>({x:r[0],y:r[1]})),css('--s1')),line('Fill P&L',T.map(r=>({x:r[0],y:r[2]})),css('--s2')),
 line('Net',T.map(r=>({x:r[0],y:r[3]})),css('--s3'))]},
 options:(o=>{o.plugins.tooltip.callbacks.label=i=>` ${i.dataset.label}: ${usd(i.parsed.y,1)}`;return o})(base(v=>usd(v)))})}
function px(){const m=D.markets.find(m=>m.cond===sel)||D.markets[0];if(!m)return;sel=m.cond;
 document.getElementById('pxh').textContent='Quotes against the book: '+m.q;
 const P=m.px,pt=(i)=>P.map(r=>({x:r[0],y:r[i]}));
 const fills=s=>m.fills.filter(f=>f[2]===s).map(f=>({x:f[0],y:f[1],q:f[3],how:f[4]}));
 draw('px',{type:'line',data:{datasets:[
  line('Mid',pt(1),css('--mid'),{borderWidth:1.5}),
  line('My bid',pt(4),css('--s1'),{stepped:true,spanGaps:false}),line('My ask',pt(5),css('--s2'),{stepped:true,spanGaps:false}),
  {type:'scatter',label:'Bought',data:fills('bid'),backgroundColor:css('--s3'),borderColor:css('--surface-1'),borderWidth:2,pointRadius:6,pointHoverRadius:8},
  {type:'scatter',label:'Sold',data:fills('ask'),backgroundColor:css('--bad'),borderColor:css('--surface-1'),borderWidth:2,pointRadius:6,pointHoverRadius:8}]},
  options:(o=>{o.plugins.tooltip.callbacks.label=i=>{const r=i.raw;return ` ${i.dataset.label}: ${i.parsed.y==null?'—':i.parsed.y.toFixed(3)}`+(r.q?` × ${r.q.toFixed(1)} (${r.how})`:'')};return o})(base(v=>v.toFixed(2)))})}
function tables(){
 document.getElementById('mk').innerHTML='<thead><tr><th>Market</th><th>Mid</th><th>Book</th><th>My quotes</th><th>Share</th><th>Quoted</th><th>Rewards</th><th>Held</th></tr></thead><tbody>'+
 D.markets.map(m=>`<tr class="click ${m.cond===sel?'sel':''}" data-c="${m.cond}"><td>${m.q} ${m.current?'':'<span class="tag">swapped out</span>'}</td>
 <td>${m.mid==null?'—':m.mid.toFixed(3)}</td><td>${m.bid==null?'—':m.bid.toFixed(3)} / ${m.ask==null?'—':m.ask.toFixed(3)}</td>
 <td>${m.my_bid==null?'—':m.my_bid.toFixed(3)} / ${m.my_ask==null?'—':m.my_ask.toFixed(3)}</td>
 <td>${(100*m.share).toFixed(1)}%</td><td>${(100*m.quoted).toFixed(0)}%</td><td>${usd(m.reward)}</td><td>${m.net.toFixed(0)}</td></tr>`).join('')+'</tbody>';
 document.querySelectorAll('#mk tbody tr').forEach(r=>r.onclick=()=>{sel=r.dataset.c;tables();px()});
 document.getElementById('fl').innerHTML=D.fills.length?'<thead><tr><th>Market</th><th>Time</th><th>Side</th><th>Price</th><th>Shares</th><th>How</th><th>Held after</th></tr></thead><tbody>'+
 D.fills.map(f=>`<tr><td>${f.q.slice(0,40)}</td><td>${tfmt(f.ts)}</td><td>${f.side==='bid'?'Buy':'Sell'}</td><td>${f.price.toFixed(3)}</td><td>${f.qty.toFixed(1)}</td><td>${f.how}</td><td>${f.net.toFixed(0)}</td></tr>`).join('')+'</tbody>'
 :'<tbody><tr><td class="sub">No fills yet.</td></tr></tbody>';
 const lg=document.getElementById('log');lg.textContent=D.log.join('\n');lg.scrollTop=lg.scrollHeight;}
async function load(){try{const r=await fetch('data',{cache:'no-store'});const j=await r.json();
 if(j.error){document.getElementById('sub').textContent=j.error;return}D=j;
 document.getElementById('sub').textContent=`${D.markets.filter(m=>m.current).length} market(s) quoting · paper run, nothing is sent to the exchange`;
 document.getElementById('upd').textContent='last sample '+tfmt(D.updated)+' · refreshes every 30 s';
 tiles();pnl();px();tables();}catch(e){document.getElementById('sub').textContent='cannot reach the dashboard server: '+e}}
load();setInterval(load,30000);
matchMedia('(prefers-color-scheme: dark)').addEventListener('change',()=>D&&(pnl(),px()));
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "data", "rewards_paper"))
    ap.add_argument("--log", default=os.path.join(HERE, "logs", "reward_paper.log"))
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/data"):
                try:
                    body, kind = json.dumps(data(a.out, a.log)).encode(), "application/json"
                except Exception as e:                        # noqa: BLE001  (a file mid-write: the next refresh works)
                    body, kind = json.dumps(dict(error=f"cannot read the run: {type(e).__name__}: {e}")).encode(), "application/json"
            else:
                body, kind = PAGE.encode(), "text/html; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *x):
            pass

    print(f"dashboard for {a.out} on http://127.0.0.1:{a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
