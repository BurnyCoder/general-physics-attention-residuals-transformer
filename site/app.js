/* Global: interactive display of saved study evidence. Local: Plotly renders data; it never changes statistical estimates. */
/* Sources: https://plotly.com/javascript/plotlyjs-function-reference/ and https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API. */
"use strict"; // Local: surface accidental global assignments instead of hiding interface bugs.
let study, selectedRollout; // Local: preserve one loaded manifest and the currently selected field case.
const el = id => document.getElementById(id); // Local: reuse a small DOM accessor for explicit element IDs.
const colors = ["#087d7c", "#d57542", "#5875b5", "#a35c93"]; // Local: stable contrasting colors identify comparable series.
const layout = title => ({title:{text:title,font:{size:16}},margin:{t:50,l:65,r:20,b:65},paper_bgcolor:"white",plot_bgcolor:"white",font:{family:"system-ui",color:"#182b39"},legend:{orientation:"h",y:-.25}}); // Local: consistent chart geometry keeps axes readable.
const draw = (id, traces, options) => Plotly.react(id, traces, options, {responsive:true,displaylogo:false}); // Global: a single plotting wrapper updates existing charts without creating duplicate canvases.
function empty(id, message) { draw(id, [], {...layout(""),xaxis:{visible:false},yaxis:{visible:false},annotations:[{text:message,showarrow:false,x:.5,y:.5,xref:"paper",yref:"paper"}]}); } // Global: missing measurements are explicit, never synthetic curves.
function options(id, values, label = value => value) { el(id).replaceChildren(...values.map(value => { const option=document.createElement("option"); option.value=value; option.textContent=label(value); return option; })); el(id).disabled=!values.length; } // Local: textContent avoids interpreting artifact labels as HTML.
function showErrors() { // Global: display full-test outcomes separately from diagnostic subsets.
  const rows=study.metrics.filter(row=>row.dataset===el("dataset").value && row.horizon===Number(el("horizon").value) && !row.intervention); // Local: retain the primary comparison's population.
  if (!rows.some(row=>Number.isFinite(row.nmse))) return empty("errors","No finite final-checkpoint measurements published yet."); // Local: never plot missing values as zero.
  draw("errors",[{x:rows.map(row=>row.run),y:rows.map(row=>row.nmse),mode:"markers",type:"scatter",marker:{size:10,color:colors[0]},name:"Measured NMSE"}],{...layout(`${el("horizon").value}-step autoregressive error`),yaxis:{title:{text:"Trajectory-averaged NMSE"},rangemode:"tozero"},xaxis:{tickangle:-35}}); // Local: zero baseline makes error magnitudes comparable.
} // Local: end the result-only display phase.
function showRun() { // Global: one selected experiment drives training, cost, and routing views.
  const name=el("run").value, rows=study.curves[name]||[], metric=el("curve").value; // Local: use the requested run without selecting by performance.
  const status=study.runs.find(row=>row.run===name); // Local: failed and pending runs remain selectable.
  el("run-status").textContent=status?`${status.status} · ${status.updates} completed updates · ${status.gpu_phase_seconds===null?"cost pending":(status.gpu_phase_seconds/3600).toFixed(3)+" GPU-phase hours"}`:"No experiment selected."; // Local: do not turn missing cost into zero training time.
  if(rows.length) draw("learning",[{x:rows.map(row=>row.step),y:rows.map(row=>row[metric]),mode:"lines",line:{color:colors[0]},name}],{...layout(`${name}: fixed 50-update means`),xaxis:{title:{text:"Optimizer updates"}},yaxis:{title:{text:metric},type:metric==="seconds"?"linear":"log"}}); else empty("learning","This experiment has no recorded training updates."); // Global: show all observed training intervals through fixed bins.
  const route=study.routing[name]; // Local: only actual diagnostic weights can appear.
  if(route) { const width=Math.max(...route.trace.map(row=>row.sources)); const z=route.trace.map(row=>Array.from({length:width},(_,index)=>row.weights[index]??null)); draw("routing",[{z,type:"heatmap",zmin:0,zmax:1,colorscale:"Viridis",colorbar:{title:{text:"Weight"}}}],{...layout(`Mean routing weights · forecast step ${route.final_forecast_step}`),xaxis:{title:{text:"Available source index (0 = embedding)"}},yaxis:{title:{text:"Read site"},autorange:"reversed"}}); } else empty("routing","No routing trace for this experiment."); // Local: unavailable future sources stay blank.
} // Local: end the experiment inspection phase.
function showInterventions() { // Global: compare within-model interventions on matched trajectories only.
  const rows=study.interventions.filter(row=>row.horizon===12); // Local: show the diagnostic analogue of the primary horizon.
  if(!rows.length) return empty("interventions","Matched intervention results are pending."); // Local: no fabricated treatment effects.
  const traces=["uniform","shuffle"].map((kind,index)=>{const selected=rows.filter(row=>row.intervention===kind); return {x:selected.map(row=>`${row.run} / ${row.dataset}`),y:selected.map(row=>row.ratio),name:kind,mode:"markers",marker:{size:9,color:colors[index]}};}); // Local: each marker remains an actual trained-model/dataset measurement.
  draw("interventions",traces,{...layout("Intervened / original NMSE on the same diagnostic subset"),yaxis:{title:{text:"Error ratio"},rangemode:"tozero"},xaxis:{tickangle:-35},shapes:[{type:"line",xref:"paper",x0:0,x1:1,y0:1,y1:1,line:{dash:"dot",color:"#777"}}]}); // Local: ratio one explicitly marks no change.
} // Local: end matched diagnostic rendering.
async function loadRollout() { // Global: fetch only the chosen actual forecast array.
  const item=study.rollouts.find(row=>row.name===el("rollout").value); // Local: URLs come from the generated allowlisted manifest.
  if(!item) { empty("truth","No saved evaluation fields yet."); empty("prediction","Predictions will appear after evaluation."); return; } // Local: handle the legitimate pre-results state.
  const response=await fetch(item.file); if(!response.ok) throw new Error(`Cannot load rollout (${response.status})`); // Local: failed requests remain visible.
  selectedRollout=await response.json(); showFields(); // Local: update both aligned field panels from the same artifact.
} // Local: end lazy field loading.
function showFields() { // Global: identical target-derived scales permit honest amplitude comparison.
  el("frame-label").textContent=el("frame").value; if(!selectedRollout) return; // Local: preserve a readable one-based frame label.
  const step=Number(el("frame").value)-1, finite=selectedRollout.truth.flat(2).filter(Number.isFinite); // Local: target data define one scale across all 24 frames.
  const low=finite.reduce((a,b)=>Math.min(a,b),Infinity), high=finite.reduce((a,b)=>Math.max(a,b),-Infinity); // Local: reductions avoid huge-array argument expansion.
  for(const [id,title] of [["truth","Ground truth"],["prediction","Autoregressive prediction"]]) draw(id,[{z:selectedRollout[id][step],type:"heatmap",zmin:low,zmax:high,colorscale:"Viridis",showscale:true}],{...layout(`${title}: ${selectedRollout.field}, step ${step+1}`),yaxis:{autorange:"reversed",scaleanchor:"x"},xaxis:{title:{text:"Stored spatial grid"}}}); // Local: actual 128x64 cells, common scale, blank nonfinite values.
} // Local: end field rendering.
async function initialize() { // Global: load the published evidence before presenting any numerical claim.
  const response=await fetch("data/study.json"); if(!response.ok) throw new Error(`Cannot load study (${response.status})`); study=await response.json(); // Local: explicit HTTP checks distinguish missing deployment data.
  const completed=study.runs.filter(row=>row.status==="complete").length; // Local: a launched process is not a completed experiment.
  el("status").textContent=`${completed} / 18 planned training runs complete. ${study.primary.status==="complete"?"Primary analysis available.":"Confirmatory results pending or undefined; no improvement is established."}`; // Global: availability and claims remain aligned.
  el("budget").textContent=`Recorded GPU time: ${study.gpu_hours.toFixed(3)} / 24 hours.`; el("updated").textContent=`Published snapshot: ${study.updated}. This public page is a saved snapshot, not a live training monitor.`; // Local: disclose staleness explicitly.
  const p=study.primary; el("effect").textContent=p.status==="complete"?`Block Attention Residuals reduce error by ${(100*p.improvement).toFixed(2)}% versus GPhyT without Attention Residuals; 95% CI [${p.improvement_ci95.map(v=>(100*v).toFixed(2)).join(", ")}]%; exact paired p=${p.permutation_p.toFixed(5)} (six pairs).`:"The primary effect and confidence interval require all six finite paired endpoints."; // Global: effect size names the matched trained baseline and never appears without uncertainty.
  for (const [mode,label] of [["standard","GPhyT without Attention Residuals"],["block","GPhyT + Block Attention Residuals"]]) { // Global: the primary table contains the same six-seed comparison as the headline.
    const rows=study.datasets.map(dataset=>(study.accuracy||[]).find(row=>row.mode===mode && row.dataset===dataset && row.horizon===12)); // Local: use existing paper aggregates without recomputing means in JavaScript.
    const tr=document.createElement("tr"), values=[label,rows[0]?`${rows[0].n_finite}/${rows[0].n_planned}`:"pending",...rows.map(row=>Number.isFinite(row?.mean)?row.mean.toLocaleString("en-US",{maximumFractionDigits:5}):"undefined")]; // Global: missing outcomes remain undefined, never zero or a reduced seed set.
    for (const value of values) { const td=document.createElement("td"); td.textContent=value; tr.append(td); } el("baseline").append(tr); // Local: textContent keeps artifact-derived display text inert.
  } // Local: baseline-first ordering is stable across page loads.
  if(study.comparison_image) { el("comparison-image").src=study.comparison_image; el("comparison").hidden=false; } // Global: show the all-model example only when the exporter has actual measured fields.
  options("dataset",study.datasets); options("run",study.runs.map(row=>row.run)); options("rollout",study.rollouts.map(row=>row.name)); // Local: only actually planned/exported identifiers enter controls.
  for(const row of study.runs) { const tr=document.createElement("tr"); for(const value of [row.run,row.status,row.updates,row.gpu_phase_seconds===null?"pending":(row.gpu_phase_seconds/3600).toFixed(3)]) { const td=document.createElement("td"); td.textContent=value; tr.append(td); } el("runs").append(tr); } // Global: every planned seed remains visible, including failures.
  const traces=[.05,.1,.2].map((effect,index)=>{const rows=study.power.filter(row=>row.reduction===effect); return {x:rows.map(row=>row.seed_log_sd),y:rows.map(row=>row.sign_flip_power),mode:"lines+markers",name:`${100*effect}% assumed improvement`,line:{color:colors[index]}};}); // Local: prospective power curves retain their explicit effect assumptions.
  draw("power-chart",traces,{...layout("Exact-test power with six independent pairs"),xaxis:{title:{text:"Assumed SD of seed log-error ratio"}},yaxis:{title:{text:"Rejection probability"},range:[0,1]}}); // Global: probability axes use the full natural range.
  el("dataset").addEventListener("change",showErrors); el("horizon").addEventListener("change",showErrors); el("run").addEventListener("change",showRun); el("curve").addEventListener("change",showRun); // Local: native inputs update only their relevant charts.
  el("rollout").addEventListener("change",()=>loadRollout().catch(fail)); el("frame").addEventListener("input",showFields); // Local: errors from asynchronous field loads are visible.
  showErrors(); showRun(); showInterventions(); await loadRollout(); // Local: initial charts reflect the same selected controls.
} // Local: end the single initialization entry point.
function fail(error) { el("status").textContent=`Viewer could not load a published artifact: ${error.message}`; console.error(error); } // Global: failures cannot masquerade as zero error or successful loading.
initialize().catch(fail); // Global: run one explicit evidence-loading pipeline.
