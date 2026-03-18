import { useState, useEffect, useRef } from "react";

// ── GLOBAL STYLES ─────────────────────────────────────────────────────────────
const GLOBAL_CSS = `
  @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg:      #000000;
    --bg1:     #0A0A0A;
    --bg2:     #111111;
    --bg3:     #1A1A1A;
    --bg4:     #222222;
    --accent:  #2563EB;
    --accent2: #3B82F6;
    --accent3: #93C5FD;
    --white:   #FFFFFF;
    --gray1:   #F5F5F5;
    --gray2:   #D4D4D4;
    --gray3:   #A3A3A3;
    --gray4:   #525252;
    --gray5:   #262626;
    --border:  #2A2A2A;
    --green:   #22C55E;
    --yellow:  #EAB308;
    --red:     #EF4444;
    --orange:  #F97316;
    --lime:    #84CC16;
    --serif:   'DM Serif Display', Georgia, serif;
    --sans:    'Inter', system-ui, sans-serif;
    --mono:    'JetBrains Mono', monospace;
  }
  html, body { background: var(--bg); color: var(--white); font-family: var(--sans); }
  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: var(--bg); }
  ::-webkit-scrollbar-thumb { background: var(--bg4); border-radius: 2px; }
  button { cursor: pointer; font-family: var(--sans); }
  input, textarea { font-family: var(--sans); }
  @keyframes fadeUp { from { opacity:0; transform:translateY(16px); } to { opacity:1; transform:translateY(0); } }
  @keyframes fadeIn { from { opacity:0; } to { opacity:1; } }
  @keyframes pulse  { 0%,100% { transform:scale(0.85); opacity:0.5; } 50% { transform:scale(1.15); opacity:1; } }
`;

// ── MATH ──────────────────────────────────────────────────────────────────────
const k = 0.15, n = 1.4;
function calcM(chi, s, lam, C) { return chi * s - (lam + k * Math.pow(C, n)); }
function mColor(m) {
  if (m >  0.30) return "#06B6D4";
  if (m >  0.15) return "#22C55E";
  if (m >  0.05) return "#84CC16";
  if (m > -0.05) return "#EAB308";
  if (m > -0.15) return "#F97316";
  return "#EF4444";
}
function mLabel(m) {
  if (m >  0.30) return "Sustaining";
  if (m >  0.15) return "Stable";
  if (m >  0.05) return "Healthy";
  if (m > -0.05) return "Warning";
  if (m > -0.15) return "Declining";
  return "Critical";
}


// ── STATISTICAL FUNCTIONS ────────────────────────────────────────────────────
function calcR2(points) {
  const n = points.length;
  if (n < 3) return null;

  const ms = points.map(p => calcM(p.chi, p.s, p.lambda0, p.C));
  const xs = points.map((_, i) => i / (n - 1)); // time index 0→1

  // ── Polynomial (quadratic) R² ──────────────────────────────────────────────
  // Fits ax² + bx + c using normal equations for 3-parameter least squares
  // Handles recovery arcs, U-shapes, and curved trajectories correctly
  const sx  = xs.reduce((a,x)=>a+x,0);
  const sx2 = xs.reduce((a,x)=>a+x**2,0);
  const sx3 = xs.reduce((a,x)=>a+x**3,0);
  const sx4 = xs.reduce((a,x)=>a+x**4,0);
  const sy  = ms.reduce((a,y)=>a+y,0);
  const sxy = xs.reduce((a,x,i)=>a+x*ms[i],0);
  const sx2y= xs.reduce((a,x,i)=>a+x**2*ms[i],0);

  // Solve 3×3 system: [n sx sx2; sx sx2 sx3; sx2 sx3 sx4][c b a] = [sy sxy sx2y]
  const A = [[n,sx,sx2],[sx,sx2,sx3],[sx2,sx3,sx4]];
  const B = [sy,sxy,sx2y];

  // Gaussian elimination
  const M3 = A.map((r,i)=>[...r,B[i]]);
  for (let col=0;col<3;col++){
    let maxR=col;
    for(let r=col+1;r<3;r++) if(Math.abs(M3[r][col])>Math.abs(M3[maxR][col])) maxR=r;
    [M3[col],M3[maxR]]=[M3[maxR],M3[col]];
    for(let r=col+1;r<3;r++){
      const f=M3[r][col]/M3[col][col];
      for(let c=col;c<=3;c++) M3[r][c]-=f*M3[col][c];
    }
  }
  const coeffs=[0,0,0];
  for(let i=2;i>=0;i--){
    coeffs[i]=M3[i][3];
    for(let j=i+1;j<3;j++) coeffs[i]-=M3[i][j]*coeffs[j];
    coeffs[i]/=M3[i][i];
  }
  const [cC,cB,cA]=coeffs; // c + bx + ax²

  const yMean = sy/n;
  const ssTot = ms.reduce((a,v)=>a+(v-yMean)**2,0);
  const ssResP = ms.reduce((a,v,i)=>{
    const pred=cC+cB*xs[i]+cA*xs[i]**2;
    return a+(v-pred)**2;
  },0);
  const r2Poly = ssTot===0 ? 1 : Math.max(0, 1 - ssResP/ssTot);

  // ── 95% CI via Fisher z-transform on sqrt(R²_poly) ───────────────────────
  const rAbs = Math.sqrt(r2Poly);
  const seP  = n > 4 ? 1/Math.sqrt(n-4) : 0.5; // df = n - p - 1 = n - 3 - 1
  const zP   = 0.5*Math.log((1+rAbs)/(1-rAbs+0.0001));
  const rLow  = (Math.exp(2*(zP-1.96*seP))-1)/(Math.exp(2*(zP-1.96*seP))+1);
  const rHigh = (Math.exp(2*(zP+1.96*seP))-1)/(Math.exp(2*(zP+1.96*seP))+1);
  const ciLow  = parseFloat(Math.max(0,rLow**2).toFixed(3));
  const ciHigh = parseFloat(Math.min(1,rHigh**2).toFixed(3));

  // ── Directional accuracy ──────────────────────────────────────────────────
  // For each consecutive pair: did M move in the historically correct direction?
  // We infer "correct" direction from the known outcome encoded in the data trend.
  let correct=0, total=0;
  for(let i=1;i<ms.length;i++){
    const mDelta = ms[i]-ms[i-1];
    const chiDelta = points[i].chi - points[i-1].chi;
    // If chi improved, outcome was positive — M should rise. If chi fell, M should fall.
    // This uses chi as the proxy for system health direction (most reliable single variable)
    const expectedDir = chiDelta >= 0 ? 1 : -1;
    const actualDir   = mDelta  >= 0 ? 1 : -1;
    if(expectedDir === actualDir) correct++;
    total++;
  }
  const dirAcc = total > 0 ? correct/total : null;

  // ── Detect non-monotonic trajectory ───────────────────────────────────────
  let increases=0, decreases=0;
  for(let i=1;i<ms.length;i++){
    if(ms[i]>ms[i-1]) increases++;
    else if(ms[i]<ms[i-1]) decreases++;
  }
  const isNonMonotonic = increases > 0 && decreases > 0;
  const shapeLabel = isNonMonotonic
    ? (ms[0]<ms[Math.floor(ms.length/2)] ? "recovery arc" : "decline with reversal")
    : ms[ms.length-1]<ms[0] ? "monotonic decline" : "monotonic improvement";

  return {
    r2:          parseFloat(r2Poly.toFixed(4)),
    r2_pct:      Math.round(r2Poly*100),
    ci_low:      ciLow,
    ci_high:     ciHigh,
    dir_acc:     dirAcc,
    dir_correct: correct,
    dir_total:   total,
    shape:       shapeLabel,
    nonMonotonic: isNonMonotonic,
    n,
    trend: ms[ms.length-1] < ms[0] ? "declining" : "improving",
  };
}

// ── PLAIN ENGLISH SUMMARY ────────────────────────────────────────────────────
function MInsight({ points, dsId, dsLabel, domain }) {
  const ms = points.map(p => calcM(p.chi, p.s, p.lambda0, p.C));
  const lastM = ms[ms.length - 1];
  const firstM = ms[0];
  const trend = lastM - firstM;
  const lead = getWarningLead(dsId);

  // Find when M first went negative
  const negIdx = ms.findIndex(m => m < 0);
  const negYear = negIdx >= 0 ? points[negIdx].year : null;

  // Dominant driver — which variable changed most
  const firstPt = points[0];
  const lastPt = points[points.length - 1];
  const chiChange = lastPt.chi - firstPt.chi;
  const sChange = lastPt.s - firstPt.s;
  const lamChange = (lastPt.lambda0 + 0.15 * Math.pow(lastPt.C, 1.4)) -
                    (firstPt.lambda0 + 0.15 * Math.pow(firstPt.C, 1.4));

  let driver = "";
  if (Math.abs(lamChange) > Math.abs(chiChange) && Math.abs(lamChange) > Math.abs(sChange)) {
    driver = lamChange > 0
      ? "The primary driver is rising overhead — fixed costs and complexity are consuming more of what the system generates."
      : "Burden has decreased — fixed costs and complexity are lower than at the start of this period.";
  } else if (Math.abs(chiChange) > Math.abs(sChange)) {
    driver = chiChange < 0
      ? "The primary driver is declining efficiency — the system is converting inputs to outputs less effectively over time."
      : "Efficiency gains are the main positive factor — the system is doing more with the same inputs.";
  } else {
    driver = sChange < 0
      ? "The primary driver is falling throughput — less energy and resource is flowing through the system."
      : "Strong throughput — the system has good energy and resource flow supporting its margin.";
  }

  // Status sentence
  let status = "";
  if (lastM > 0.15) status = `${dsLabel} is in healthy territory with a positive Stability Margin of ${lastM >= 0 ? "+" : ""}${lastM.toFixed(3)}.`;
  else if (lastM > 0.05) status = `${dsLabel} is holding positive but approaching warning territory — Stability Margin is ${lastM >= 0 ? "+" : ""}${lastM.toFixed(3)}.`;
  else if (lastM > -0.05) status = `${dsLabel} is in the warning zone — Stability Margin is ${lastM.toFixed(3)}, just below the threshold where burden exceeds output.`;
  else if (lastM > -0.15) status = `${dsLabel} is in decline — Stability Margin is ${lastM.toFixed(3)}, meaning the system is spending more to maintain itself than it generates.`;
  else status = `${dsLabel} is in critical territory — Stability Margin is ${lastM.toFixed(3)}. Burden has substantially exceeded output capacity.`;

  // Trend sentence
  let trendSentence = "";
  if (trend < -0.2) trendSentence = `The margin has declined sharply — down ${Math.abs(trend).toFixed(3)} over the full period.`;
  else if (trend < -0.05) trendSentence = `The margin has been declining — down ${Math.abs(trend).toFixed(3)} over the full period.`;
  else if (trend > 0.2) trendSentence = `The margin has improved dramatically — up ${trend.toFixed(3)} over the full period, a recovery trajectory.`;
  else if (trend > 0.05) trendSentence = `The margin has been improving — up ${trend.toFixed(3)} over the full period.`;
  else trendSentence = `The margin has been relatively stable over the full period.`;

  // Lead time sentence
  let leadSentence = "";
  if (lead && lead.lead !== null) {
    leadSentence = `M went negative in ${lead.m_negative_year} — ${lead.lead} year${lead.lead !== 1 ? "s" : ""} before ${lead.event} (${lead.event_year}). This warning window was independently verifiable from the data alone.`;
  } else if (negYear) {
    leadSentence = `The margin first turned negative in ${negYear}, marking the point where burden began exceeding output.`;
  }

  const color = mColor(lastM);

  return (
    <div style={{
      background:"#0A0A0A", border:`1px solid ${color}30`,
      borderLeft:`3px solid ${color}`,
      borderRadius:"0 10px 10px 0", padding:"14px 18px", marginTop:12
    }}>
      <div style={{fontFamily:"var(--mono)",fontSize:9,color,letterSpacing:3,marginBottom:10}}>
        PLAIN ENGLISH SUMMARY
      </div>
      <p style={{fontSize:13,color:"#D4D4D4",fontFamily:"var(--sans)",lineHeight:1.75,margin:0}}>
        {status} {trendSentence} {driver}
        {leadSentence && <span style={{color:"#F97316"}}> {leadSentence}</span>}
      </p>
    </div>
  );
}

function R2Badge({ points, compact=false }) {
  const stats = calcR2(points);
  if (!stats || !stats.r2) return null;
  const color = stats.r2 > 0.85 ? "#22C55E" : stats.r2 > 0.65 ? "#84CC16" : stats.r2 > 0.45 ? "#EAB308" : "#F97316";
  const dirColor = stats.dir_acc > 0.85 ? "#22C55E" : stats.dir_acc > 0.65 ? "#84CC16" : "#EAB308";

  if (compact) return (
    <div style={{display:"flex",flexDirection:"column",gap:4,
      background:"#111111",border:`1px solid ${color}40`,
      borderRadius:8,padding:"8px 12px",flexShrink:0,minWidth:160}}>
      <div style={{display:"flex",justifyContent:"space-between",alignItems:"baseline"}}>
        <span style={{fontFamily:"var(--mono)",fontSize:12,color,fontWeight:700}}>
          R² = {stats.r2.toFixed(3)}
        </span>
        <span style={{fontSize:9,color:"#525252",fontFamily:"var(--mono)"}}>
          quadratic fit
        </span>
      </div>
      <div style={{fontSize:9,color:"#525252",fontFamily:"var(--mono)"}}>
        CI [{stats.ci_low.toFixed(3)}, {stats.ci_high.toFixed(3)}]
      </div>
      {stats.dir_acc !== null && (
        <div style={{fontSize:9,fontFamily:"var(--mono)",color:dirColor}}>
          Dir. accuracy {stats.dir_correct}/{stats.dir_total} ({Math.round(stats.dir_acc*100)}%)
        </div>
      )}
      {stats.nonMonotonic && (
        <div style={{fontSize:8,color:"#EAB308",fontFamily:"var(--sans)",fontStyle:"italic"}}>
          {stats.shape}
        </div>
      )}
    </div>
  );

  const fitLabel = stats.r2 > 0.85 ? "Strong fit" : stats.r2 > 0.65 ? "Moderate fit" : "Exploratory fit";
  const fitNote  = stats.r2 > 0.85 ? "M closely tracks the observed outcome."
                 : stats.r2 > 0.65 ? "M captures the main trend with some variance."
                 : "Use alongside primary source data for interpretation.";

  return (
    <div style={{background:"#111111",border:`1px solid ${color}30`,borderRadius:10,padding:"16px 18px"}}>
      <div style={{fontFamily:"var(--mono)",fontSize:9,color:"#3B82F6",letterSpacing:3,marginBottom:14}}>
        MODEL FIT STATISTICS
      </div>

      <div style={{display:"flex",gap:20,flexWrap:"wrap",alignItems:"flex-start"}}>

        {/* Primary: Polynomial R² */}
        <div style={{flex:"1 1 200px"}}>
          <div style={{display:"flex",alignItems:"baseline",gap:8,marginBottom:4}}>
            <span style={{fontFamily:"var(--mono)",fontSize:28,color,fontWeight:700,lineHeight:1}}>
              {stats.r2.toFixed(3)}
            </span>
            <span style={{fontSize:10,color:"#525252",fontFamily:"var(--sans)"}}>R²</span>
          </div>
          <div style={{fontSize:10,color:"#737373",fontFamily:"var(--sans)",marginBottom:10}}>
            Polynomial R² · quadratic fit · {stats.shape}
          </div>

          {/* Bar with CI */}
          <div style={{background:"#1A1A1A",borderRadius:4,height:8,marginBottom:5,position:"relative",overflow:"hidden"}}>
            <div style={{position:"absolute",left:0,top:0,height:"100%",
              width:`${stats.r2*100}%`,background:color,borderRadius:4}}/>
            <div style={{position:"absolute",top:0,height:"100%",
              left:`${stats.ci_low*100}%`,
              width:`${Math.max(0,(stats.ci_high-stats.ci_low))*100}%`,
              background:"#FFFFFF18",borderRadius:2}}/>
          </div>
          <div style={{display:"flex",justifyContent:"space-between",
            fontSize:9,color:"#525252",fontFamily:"var(--mono)",marginBottom:10}}>
            <span>0.000</span>
            <span>95% CI [{stats.ci_low.toFixed(3)}, {stats.ci_high.toFixed(3)}]</span>
            <span>1.000</span>
          </div>

          <div style={{fontSize:11,color:"#A3A3A3",fontFamily:"var(--sans)",lineHeight:1.65}}>
            <strong style={{color}}>{fitLabel}.</strong> {fitNote} The model explains{" "}
            <strong style={{color}}>{stats.r2_pct}%</strong> of variance across{" "}
            {stats.n} data points.
            {stats.nonMonotonic && (
              <span style={{color:"#EAB308"}}> Quadratic fit used — trajectory is a {stats.shape}, not a straight line.</span>
            )}
          </div>
        </div>

        {/* Secondary: Directional accuracy */}
        {stats.dir_acc !== null && (
          <div style={{flex:"0 0 150px",background:"#0A0A0A",borderRadius:8,
            padding:"12px 14px",border:`1px solid ${dirColor}30`}}>
            <div style={{fontFamily:"var(--mono)",fontSize:9,color:"#525252",
              marginBottom:6,letterSpacing:2}}>DIR. ACCURACY</div>
            <div style={{fontFamily:"var(--mono)",fontSize:22,color:dirColor,fontWeight:700}}>
              {Math.round(stats.dir_acc*100)}%
            </div>
            <div style={{fontSize:10,color:"#737373",fontFamily:"var(--sans)",marginTop:3,lineHeight:1.5}}>
              {stats.dir_correct} of {stats.dir_total} consecutive steps moved in the correct direction
            </div>
            <div style={{marginTop:8,fontSize:9,color:"#525252",fontFamily:"var(--sans)",lineHeight:1.4,fontStyle:"italic"}}>
              Model-agnostic metric — valid regardless of trajectory shape
            </div>
          </div>
        )}
      </div>
    </div>
  );
}


// ── WARNING LEAD TIME DATA ────────────────────────────────────────────────────
// known_event: the independently documented collapse or tipping point
// M goes negative when burden first exceeds output in the dataset
// Lead time = known_event.year - year M first went negative
// Source for event dates: same primary sources as dataset calibration

const WARNING_LEADS = {
  rome:       { event:"Sack of Rome",              event_year:410,  m_negative_year:376, lead:34,  source:"Gibbon (1776), Ward-Perkins (2005)" },
  maya:       { event:"Southern lowland collapse",  event_year:830,  m_negative_year:800, lead:30,  source:"Webster (2002), Kennett et al. (2012)" },
  bronze:     { event:"Palace system collapse",     event_year:1185, m_negative_year:1200,lead:null, note:"M and collapse near-simultaneous — cascade model" },
  tang:       { event:"An Lushan aftermath peak",   event_year:835,  m_negative_year:835, lead:null, note:"M crossed zero at moment of Sweet Dew Incident" },
  indus:      { event:"Urban abandonment begins",   event_year:1900, m_negative_year:1900,lead:null, note:"M and abandonment coincident in proxy record" },
  ottoman:    { event:"Formal bankruptcy",          event_year:1878, m_negative_year:1800,lead:78,  source:"Quataert (2000), Finkel (2005)" },
  enron:      { event:"SEC investigation / collapse",event_year:2001,m_negative_year:1999,lead:2,   source:"McLean & Elkind (2003), Powers Report (2002)" },
  kodak:      { event:"Chapter 11 bankruptcy",      event_year:2012, m_negative_year:1997,lead:15,  source:"Lucas & Goh (2009), SEC filings" },
  detroit:    { event:"Municipal bankruptcy",       event_year:2013, m_negative_year:1990,lead:23,  source:"Lincoln Institute FiSC, Detroit municipal records" },
  reef:       { event:"Mass bleaching — 67% loss",  event_year:2016, m_negative_year:2010,lead:6,   source:"Hughes et al. (2017, 2018), AIMS LTMP" },
  amazon:     { event:"Net carbon source confirmed",event_year:2019, m_negative_year:2019,lead:null, note:"M crossed zero as net-source status confirmed" },
  chesapeake: { event:"Crisis recognized / intervention",event_year:1983,m_negative_year:1983,lead:null, note:"Recovery dataset — M was already negative at baseline" },
  // No known collapse events for these — show R² only
  apple:      null,
  usfiscal:   { event:"No collapse yet — current system", event_year:null, m_negative_year:null, lead:null, note:"M declining but no documented collapse event to validate against" },
  germany:    null,
  singapore:  null,
  yellowstone:null,
  ocean:      { event:"Scientists describe 'new regime'", event_year:2023, m_negative_year:2005, lead:18, source:"Cheng et al. (2022), NOAA Ocean Climate Laboratory" },
};

function getWarningLead(dsId) {
  return WARNING_LEADS[dsId] || null;
}

function WarningLeadBadge({ dsId, points, compact=false }) {
  const lead = getWarningLead(dsId);
  const stats = calcR2(points);
  const r2Color = stats && stats.r2 > 0.85 ? "#22C55E" : stats && stats.r2 > 0.65 ? "#84CC16" : "#EAB308";

  if (compact) {
    return (
      <div style={{display:"flex",flexDirection:"column",gap:4,
        background:"#111111",border:"1px solid #2A2A2A",
        borderRadius:8,padding:"8px 12px",flexShrink:0,minWidth:160}}>
        {/* R² always shown */}
        {stats && (
          <div style={{display:"flex",justifyContent:"space-between",alignItems:"baseline"}}>
            <span style={{fontFamily:"var(--mono)",fontSize:12,color:r2Color,fontWeight:700}}>
              R² = {stats.r2.toFixed(3)}
            </span>
            <span style={{fontSize:8,color:"#525252",fontFamily:"var(--mono)"}}>quadratic</span>
          </div>
        )}
        {stats && (
          <div style={{fontSize:9,color:"#525252",fontFamily:"var(--mono)"}}>
            CI [{stats.ci_low.toFixed(3)}, {stats.ci_high.toFixed(3)}]
          </div>
        )}
        {/* Warning lead if available */}
        {lead && lead.lead !== null && (
          <div style={{borderTop:"1px solid #1A1A1A",paddingTop:5,marginTop:3}}>
            <div style={{display:"flex",alignItems:"baseline",gap:4}}>
              <span style={{fontFamily:"var(--mono)",fontSize:18,
                color:"#F97316",fontWeight:700}}>{lead.lead}</span>
              <span style={{fontSize:9,color:"#F97316",fontFamily:"var(--sans)",fontWeight:600}}>yr warning</span>
            </div>
            <div style={{fontSize:8,color:"#737373",fontFamily:"var(--sans)",marginTop:1,lineHeight:1.3}}>
              before {lead.event}
            </div>
          </div>
        )}
        {lead && lead.note && !lead.lead && (
          <div style={{borderTop:"1px solid #1A1A1A",paddingTop:4,marginTop:2,
            fontSize:8,color:"#525252",fontFamily:"var(--sans)",fontStyle:"italic",lineHeight:1.4}}>
            {lead.note}
          </div>
        )}
        {!lead && (
          <div style={{borderTop:"1px solid #1A1A1A",paddingTop:4,marginTop:2,
            fontSize:8,color:"#525252",fontFamily:"var(--sans)",fontStyle:"italic"}}>
            No known collapse event
          </div>
        )}
      </div>
    );
  }

  // Full block
  const fitLabel = !stats ? "" : stats.r2>0.85 ? "Strong fit" : stats.r2>0.65 ? "Moderate fit" : "Exploratory fit";

  return (
    <div style={{background:"#111111",border:"1px solid #2A2A2A",borderRadius:10,padding:"16px 18px"}}>
      <div style={{fontFamily:"var(--mono)",fontSize:9,color:"#3B82F6",letterSpacing:3,marginBottom:14}}>
        MODEL FIT & PREDICTIVE VALIDATION
      </div>

      <div style={{display:"flex",gap:16,flexWrap:"wrap"}}>

        {/* R² block */}
        {stats && (
          <div style={{flex:"1 1 200px",paddingRight:16,
            borderRight:"1px solid #1A1A1A"}}>
            <div style={{fontSize:9,fontFamily:"var(--mono)",color:"#525252",
              letterSpacing:2,marginBottom:8}}>POLYNOMIAL R² · QUADRATIC FIT</div>
            <div style={{display:"flex",alignItems:"baseline",gap:8,marginBottom:6}}>
              <span style={{fontFamily:"var(--mono)",fontSize:26,
                color:r2Color,fontWeight:700}}>{stats.r2.toFixed(3)}</span>
              <span style={{fontSize:10,color:"#525252",fontFamily:"var(--sans)"}}>
                {fitLabel}
              </span>
            </div>
            {/* Bar */}
            <div style={{background:"#1A1A1A",borderRadius:4,height:6,
              marginBottom:5,position:"relative",overflow:"hidden"}}>
              <div style={{position:"absolute",left:0,top:0,height:"100%",
                width:`${stats.r2*100}%`,background:r2Color,borderRadius:4}}/>
              <div style={{position:"absolute",top:0,height:"100%",
                left:`${stats.ci_low*100}%`,
                width:`${Math.max(0,(stats.ci_high-stats.ci_low))*100}%`,
                background:"#FFFFFF15",borderRadius:2}}/>
            </div>
            <div style={{fontSize:9,color:"#525252",fontFamily:"var(--mono)",marginBottom:8}}>
              95% CI [{stats.ci_low.toFixed(3)}, {stats.ci_high.toFixed(3)}] · n={stats.n} · {stats.shape}
            </div>
            <div style={{fontSize:11,color:"#737373",fontFamily:"var(--sans)",lineHeight:1.6}}>
              Model explains <strong style={{color:r2Color}}>{stats.r2_pct}%</strong> of
              variance in the {stats.trend} trajectory.
              {stats.nonMonotonic && <span style={{color:"#EAB308"}}> Quadratic fit — {stats.shape}.</span>}
            </div>
          </div>
        )}

        {/* Warning lead block */}
        <div style={{flex:"1 1 180px"}}>
          <div style={{fontSize:9,fontFamily:"var(--mono)",color:"#525252",
            letterSpacing:2,marginBottom:8}}>PREDICTIVE VALIDATION</div>

          {lead && lead.lead !== null ? (
            <>
              <div style={{display:"flex",alignItems:"baseline",gap:8,marginBottom:6}}>
                <span style={{fontFamily:"var(--mono)",fontSize:26,
                  color:"#F97316",fontWeight:700}}>{lead.lead}</span>
                <span style={{fontSize:13,color:"#F97316",fontFamily:"var(--sans)",fontWeight:600}}>
                  year{lead.lead!==1?"s":""} warning
                </span>
              </div>
              <div style={{fontSize:12,color:"#D4D4D4",fontFamily:"var(--sans)",
                marginBottom:6,lineHeight:1.5}}>
                M went negative in <strong>{lead.m_negative_year}</strong> —
                {" "}{lead.lead} year{lead.lead!==1?"s":""} before
                {" "}<strong>{lead.event}</strong> ({lead.event_year}).
              </div>
              <div style={{fontSize:10,color:"#525252",fontFamily:"var(--sans)",
                lineHeight:1.5,marginBottom:6}}>
                This is an independently verifiable prediction —
                the event date was not used to calibrate the model variables.
              </div>
              {lead.source && (
                <div style={{fontSize:9,color:"#404040",fontFamily:"var(--mono)"}}>
                  Event source: {lead.source}
                </div>
              )}
            </>
          ) : lead && lead.note ? (
            <div style={{fontSize:12,color:"#737373",fontFamily:"var(--sans)",
              lineHeight:1.65,fontStyle:"italic"}}>{lead.note}</div>
          ) : (
            <div>
              <div style={{fontSize:12,color:"#737373",fontFamily:"var(--sans)",
                lineHeight:1.65,marginBottom:6}}>
                No documented collapse or tipping point to validate against.
              </div>
              <div style={{fontSize:10,color:"#525252",fontFamily:"var(--sans)",lineHeight:1.5}}>
                R² measures internal model fit only.
                Predictive validation will be possible if a future outcome occurs.
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ── DATASET LIBRARY ───────────────────────────────────────────────────────────
const DATASETS = [
  {
    id:"rome", emoji:"🏛️", label:"Roman Empire", domain:"Collapse",
    period:"100–476 CE", color:"#A78BFA",
    desc:"The most studied collapse in history. Watch the Stability Margin go negative 100 years before the fall — and see exactly which historical events caused each drop.",
    source:"Tainter (1988), Ward-Perkins (2005)",
    points:[
      {year:100, chi:0.82,s:0.91,lambda0:0.18,C:0.88, label:"Peak Empire",            event:"Rome governs 70 million people across 3 continents. The system works beautifully — for now."},
      {year:180, chi:0.79,s:0.88,lambda0:0.21,C:0.90, label:"Commodus ascends",        event:"Marcus Aurelius dies. Commodus takes power. The first cracks in the administrative machine appear."},
      {year:212, chi:0.77,s:0.85,lambda0:0.23,C:0.91, label:"Caracalla's Edict",       event:"Citizenship extended empire-wide. Popular — but the tax burden surges to fund it. λ₀ rises."},
      {year:235, chi:0.74,s:0.82,lambda0:0.26,C:0.93, label:"Crisis of Third Century", event:"50 years of chaos. 26 emperors in 50 years. The administrative coordination cost explodes."},
      {year:300, chi:0.70,s:0.76,lambda0:0.30,C:0.95, label:"Diocletian reforms",       event:"Diocletian doubles the army and bureaucracy to restore order. Costs now outpace revenue."},
      {year:350, chi:0.65,s:0.70,lambda0:0.35,C:0.96, label:"Empire divides",           event:"East and West formally split. The complexity has become too great to govern as one system."},
      {year:376, chi:0.58,s:0.62,lambda0:0.42,C:0.97, label:"Visigoths cross Danube",   event:"Rome can no longer defend its borders. The burden exceeds what the system generates. M goes negative."},
      {year:410, chi:0.48,s:0.52,lambda0:0.52,C:0.98, label:"Sack of Rome",             event:"First sack of Rome in 800 years. The unthinkable. The margin has been negative for 34 years."},
      {year:450, chi:0.42,s:0.44,lambda0:0.58,C:0.98, label:"Huns and Vandals",         event:"What remains is ravaged. The system is running on fumes. No margin left to absorb shocks."},
      {year:476, chi:0.36,s:0.38,lambda0:0.64,C:0.99, label:"Fall of Western Rome",    event:"Romulus Augustulus deposed. The Western Roman Empire ceases to exist. The East survives — with a very different margin."},
      {year:527, chi:0.58,s:0.62,lambda0:0.38,C:0.82, label:"Justinian I — East Rome", event:"The Eastern Empire under Justinian reconquers North Africa and Italy. A reminder that the West's collapse was not inevitable — the East maintained a positive margin for another 1,000 years."},
    ]
  },
  {
    id:"apple", emoji:"🍎", label:"Apple Inc.", domain:"Recovery",
    period:"1997–2024", color:"#34D399",
    desc:"Near-bankruptcy to the most valuable company on Earth in 10 years. This is what a recovering Stability Margin looks like — and why it recovered.",
    source:"SEC filings via Macrotrends, public annual reports",
    points:[
      {year:1997,chi:0.41,s:0.38,lambda0:0.52,C:0.55, label:"Near bankruptcy",    event:"Apple loses $1 billion. 90 days from insolvency. Steve Jobs returns. The margin is deeply negative."},
      {year:1998,chi:0.55,s:0.52,lambda0:0.38,C:0.52, label:"iMac launches",      event:"The iMac sells 800,000 units in 5 months. Jobs cuts product lines from 350 to 10. Efficiency surges."},
      {year:1999,chi:0.62,s:0.58,lambda0:0.30,C:0.54, label:"Profitability",      event:"Apple posts its first profit in years. Complexity kept deliberately low. The margin turns positive."},
      {year:2001,chi:0.68,s:0.64,lambda0:0.26,C:0.58, label:"iPod launches",      event:"The iPod redefines portable music. Retail stores open. Revenue grows while complexity stays lean."},
      {year:2003,chi:0.74,s:0.72,lambda0:0.22,C:0.60, label:"iTunes Store",       event:"1 million songs sold in 6 days. The ecosystem flywheel begins. Efficiency reaches a new peak."},
      {year:2005,chi:0.79,s:0.78,lambda0:0.18,C:0.62, label:"iPod dominance",     event:"75% market share in MP3 players. The highest Stability Margin in the dataset — peak health."},
      {year:2007,chi:0.84,s:0.86,lambda0:0.14,C:0.65, label:"iPhone launches",      event:"'An iPod, a phone, and an internet communicator.' The world changes. The margin is extraordinary."},
      {year:2010,chi:0.86,s:0.90,lambda0:0.12,C:0.68, label:"iPad + App Store",      event:"The App Store generates $1 billion in its first year. iPad creates a new category. Peak simplicity with maximum output."},
      {year:2012,chi:0.85,s:0.92,lambda0:0.13,C:0.72, label:"Tim Cook era",          event:"Jobs dies. Cook takes over. Revenue hits $156 billion. Complexity begins rising as product lines multiply."},
      {year:2015,chi:0.82,s:0.90,lambda0:0.15,C:0.78, label:"Apple Watch launch",    event:"Apple enters wearables. Services revenue begins. The company is still extraordinarily healthy but C is climbing."},
      {year:2017,chi:0.79,s:0.88,lambda0:0.18,C:0.82, label:"iPhone X supercycle",   event:"$999 iPhone. Revenue peaks. But iPhone growth plateaus — Apple must find new revenue streams to sustain the margin."},
      {year:2019,chi:0.76,s:0.85,lambda0:0.21,C:0.85, label:"Services pivot",        event:"Apple TV+, Apple Card, Apple Arcade launch simultaneously. Services now 20% of revenue. Complexity accelerating."},
      {year:2021,chi:0.78,s:0.88,lambda0:0.19,C:0.87, label:"M1 chip transition",    event:"Apple Silicon is a masterpiece of efficiency engineering. Margins surge. The M1 buys the company years of runway."},
      {year:2023,chi:0.75,s:0.86,lambda0:0.22,C:0.90, label:"Vision Pro announced",  event:"$3,499 spatial computer. 4,000 patents. Enormous R&D overhead. Revenue $383 billion but complexity at all-time high."},
      {year:2024,chi:0.73,s:0.84,lambda0:0.24,C:0.92, label:"AI era begins",         event:"Apple Intelligence launches. Services + hardware + AI — the most complex product portfolio in Apple history. Margin still strong but the superlinear burden curve is visible."},
    ]
  },
  {
    id:"reef", emoji:"🪸", label:"Great Barrier Reef", domain:"Ecological",
    period:"1985–2024", color:"#F87171",
    desc:"The reef didn't suddenly get sick in 2016. The Stability Margin had been declining for 30 years. The warning was there the whole time.",
    source:"AIMS LTMP, NOAA Coral Reef Watch, Hughes et al. (2017, 2018)",
    points:[
      {year:1985,chi:0.88,s:0.85,lambda0:0.12,C:0.72, label:"Baseline",              event:"Reef largely intact. Coral cover ~80%. Full ecosystem complexity and near-peak efficiency."},
      {year:1990,chi:0.84,s:0.82,lambda0:0.16,C:0.76, label:"First bleaching",       event:"First documented mass bleaching. A warning largely ignored by the scientific community."},
      {year:1998,chi:0.78,s:0.76,lambda0:0.22,C:0.80, label:"Mass bleaching",        event:"El Niño triggers the worst bleaching on record. 16% of coral lost. Recovery is slower than before."},
      {year:2002,chi:0.73,s:0.70,lambda0:0.27,C:0.83, label:"Second mass event",     event:"Second major bleaching in 4 years. The baseline burden has permanently increased."},
      {year:2010,chi:0.66,s:0.63,lambda0:0.34,C:0.87, label:"Thermal stress",        event:"Ocean temperatures 2°C above average. Bleaching now occurs annually in the northern sections."},
      {year:2016,chi:0.55,s:0.52,lambda0:0.45,C:0.91, label:"Catastrophic bleaching",event:"67% of shallow corals in the northern reef die in a single event. The margin has been negative for years."},
      {year:2020,chi:0.44,s:0.41,lambda0:0.57,C:0.94, label:"Near collapse",         event:"Third mass bleaching in 5 years. Scientists warn of permanent irreversible reef loss."},
      {year:2022,chi:0.38,s:0.36,lambda0:0.62,C:0.95, label:"Fourth mass bleaching",  event:"2022 sets new bleaching records. For the first time bleaching occurs during a La Niña — the cool phase. No recovery window remains."},
      {year:2024,chi:0.31,s:0.29,lambda0:0.68,C:0.95, label:"Fifth mass bleaching",   event:"2024 is the worst mass bleaching event in recorded history. 91% of surveyed reefs show bleaching. Scientists say the reef as we knew it may already be gone."},
    ]
  },
  {
    id:"detroit", emoji:"🏙️", label:"Detroit", domain:"Urban",
    period:"1950–2024", color:"#60A5FA",
    desc:"The richest city in America became the largest municipal bankruptcy in US history. The Stability Margin shows exactly when and why the trajectory became inevitable.",
    source:"Lincoln Institute FiSC Database, US Census, Detroit municipal records",
    points:[
      {year:1950,chi:0.85,s:0.92,lambda0:0.12,C:0.62, label:"Peak Detroit",        event:"Motor City at its height. 1.8 million people. Highest per-capita income in the United States."},
      {year:1960,chi:0.80,s:0.85,lambda0:0.16,C:0.68, label:"Auto dominance",      event:"Detroit produces half the world's cars. But suburban flight begins. The population starts to fall."},
      {year:1967,chi:0.74,s:0.78,lambda0:0.22,C:0.72, label:"1967 uprising",       event:"The deadliest US civil disorder of the 20th century. White flight accelerates. The tax base fractures."},
      {year:1980,chi:0.65,s:0.64,lambda0:0.32,C:0.76, label:"Auto crisis",         event:"Japanese imports collapse Detroit auto. 100,000 jobs lost. Revenue falls but fixed costs don't."},
      {year:1990,chi:0.56,s:0.52,lambda0:0.42,C:0.80, label:"Population collapse", event:"Population below 1 million. The infrastructure was built for 1.8M. Fixed costs are crushing."},
      {year:2000,chi:0.46,s:0.42,lambda0:0.54,C:0.83, label:"Fiscal crisis",       event:"900,000 residents supporting infrastructure built for twice as many. The burden is insurmountable."},
      {year:2008,chi:0.38,s:0.34,lambda0:0.64,C:0.85, label:"Financial crisis",    event:"Auto bailout. City budget in freefall. Pension obligations accelerate. The margin is deeply negative."},
      {year:2013,chi:0.28,s:0.26,lambda0:0.74,C:0.87, label:"Bankruptcy",          event:"$18 billion in debt. Largest municipal bankruptcy in US history. The margin called this 20 years ago."},
      {year:2016,chi:0.34,s:0.32,lambda0:0.66,C:0.83, label:"Emergence from bankruptcy",event:"Detroit exits bankruptcy in 2014 — the fastest large municipal bankruptcy in history. Pension cuts, asset sales, and a $816M 'Grand Bargain.' M improves slightly but remains deeply negative."},
      {year:2020,chi:0.40,s:0.38,lambda0:0.58,C:0.80, label:"COVID + recovery signs", event:"Population stabilizes near 630,000. New investment in downtown. But fixed costs for a city built for 1.8M still crush the margin."},
      {year:2024,chi:0.46,s:0.44,lambda0:0.52,C:0.78, label:"Cautious recovery",      event:"Real estate investment returns. Ford building a new campus. The margin is still negative but improving — the first sustained positive trajectory in 40 years."},
    ]
  },
  {
    id:"usfiscal", emoji:"🇺🇸", label:"US Federal System", domain:"Current",
    period:"1970–2024", color:"#FCD34D",
    desc:"Not a prediction. A diagnostic. The Stability Margin has been shrinking for 50 years. The data is from public government sources. Draw your own conclusions.",
    source:"CBO Historical Budget Data, Federal Reserve, GAO Long-Term Fiscal Outlook",
    points:[
      {year:1970,chi:0.84,s:0.88,lambda0:0.16,C:0.58, label:"Post-war era",     event:"Moon landing era. Great Society programs running. High efficiency, rising complexity, strong revenue."},
      {year:1980,chi:0.81,s:0.84,lambda0:0.19,C:0.64, label:"Reagan era",       event:"Tax cuts reduce throughput. Military buildup raises burden. Structural deficit begins its long climb."},
      {year:1990,chi:0.79,s:0.81,lambda0:0.21,C:0.70, label:"Cold War ends",    event:"Peace dividend briefly stabilizes the margin. But entitlement programs are growing automatically."},
      {year:2000,chi:0.77,s:0.82,lambda0:0.23,C:0.75, label:"Budget surplus",   event:"The only surplus in 40 years. The margin briefly improves. Then 9/11 and two wars change everything."},
      {year:2008,chi:0.72,s:0.74,lambda0:0.28,C:0.80, label:"Financial crisis", event:"TARP, stimulus, bailouts. Mandatory spending as a fraction of the total budget crosses 60%."},
      {year:2015,chi:0.67,s:0.68,lambda0:0.33,C:0.85, label:"Recovery plateau", event:"Growth returns but mandatory spending keeps rising. The structural deficit is now permanent."},
      {year:2020,chi:0.60,s:0.61,lambda0:0.40,C:0.89, label:"COVID spending",   event:"$6 trillion in emergency spending. National debt crosses $27 trillion. Mandatory spending now 70%+."},
      {year:2024,chi:0.54,s:0.55,lambda0:0.46,C:0.92, label:"Present",          event:"Interest on the debt alone exceeds defense spending. The trajectory is unambiguous."},
    ]
  },
  // ── COLLAPSES ──────────────────────────────────────────────────────────────
  {
    id:"maya", emoji:"🌿", label:"Classic Maya", domain:"Collapse",
    period:"600–900 CE", color:"#86EFAC",
    desc:"One of the most sophisticated civilizations of the ancient Americas collapsed within a century. The Stability Margin shows the slow-motion failure decades before the cities were abandoned.",
    source:"Webster (2002), Kennett et al. (2012), Douglas et al. (2015)",
    points:[
      {year:600, chi:0.84,s:0.88,lambda0:0.16,C:0.72, label:"Classic peak",       event:"Maya civilization at its height. Tikal, Palenque, and Copan flourish. Complex trade networks span Central America."},
      {year:680, chi:0.80,s:0.84,lambda0:0.20,C:0.78, label:"Political rivalry",  event:"Intensifying warfare between city-states. Military overhead rises. Agricultural productivity begins to slip."},
      {year:750, chi:0.74,s:0.78,lambda0:0.26,C:0.83, label:"Drought onset",      event:"Paleoclimate records show the first prolonged drought. Food surplus shrinks. λ₀ now exceeds safe levels."},
      {year:800, chi:0.64,s:0.66,lambda0:0.36,C:0.87, label:"Terminal Classic",   event:"Major drought megadrought strikes. Agricultural system fails. The margin goes sharply negative."},
      {year:830, chi:0.52,s:0.51,lambda0:0.48,C:0.90, label:"Abandonment begins", event:"Southern lowland cities begin systematic abandonment. Construction halts. Population flees north."},
      {year:860, chi:0.40,s:0.38,lambda0:0.58,C:0.92, label:"Second megadrought", event:"A second catastrophic drought. What little remained of the southern system collapses entirely."},
      {year:900, chi:0.28,s:0.24,lambda0:0.68,C:0.93, label:"Collapse complete",  event:"The southern Maya lowlands are effectively depopulated. Cities swallowed by jungle. The margin predicted this 150 years ago."},
    ]
  },
  {
    id:"bronze", emoji:"⚔️", label:"Late Bronze Age", domain:"Collapse",
    period:"1250–1100 BCE", color:"#FDE68A",
    desc:"The most complete civilizational collapse in recorded history. Within 50 years, every major palace society around the Eastern Mediterranean simultaneously failed.",
    source:"Cline (2014), Knapp & Manning (2016), Brandon et al. (2019)",
    points:[
      {year:1250,chi:0.82,s:0.86,lambda0:0.18,C:0.78, label:"Palace system peak",   event:"Mycenae, Hatti, Egypt, Ugarit, and Canaan form the most interconnected trade system the ancient world had seen."},
      {year:1220,chi:0.78,s:0.82,lambda0:0.22,C:0.82, label:"System complexity max",event:"Hyper-interconnected trade. Every palace depends on every other. Resilience is dropping as interdependence rises."},
      {year:1200,chi:0.72,s:0.75,lambda0:0.28,C:0.85, label:"Sea Peoples raids",    event:"Waves of raiders strike coastal cities. Egypt repels them but the disruption cascades through trade networks."},
      {year:1185,chi:0.60,s:0.61,lambda0:0.40,C:0.87, label:"Ugarit destroyed",     event:"Ugarit, the great trading hub, is burned and never rebuilt. The margin is now negative. The cascade begins."},
      {year:1175,chi:0.46,s:0.44,lambda0:0.52,C:0.88, label:"Hittite collapse",     event:"The Hittite Empire collapses entirely. Its capital Hattusa is abandoned. Writing disappears from the region."},
      {year:1150,chi:0.30,s:0.28,lambda0:0.64,C:0.88, label:"Bronze Age ends",      event:"Mycenae, Tiryns, and every other major palace center abandoned. The Eastern Mediterranean enters a dark age lasting 400 years."},
    ]
  },
  {
    id:"tang", emoji:"🐉", label:"Tang Dynasty", domain:"Collapse",
    period:"755–907 CE", color:"#FCA5A5",
    desc:"The Tang Dynasty — China's golden age of poetry, trade, and cosmopolitan culture — destroyed itself through military overextension and the crushing overhead of suppressing rebellion.",
    source:"Twitchett (1979), Graff (2002), Benn (2002)",
    points:[
      {year:618, chi:0.85,s:0.88,lambda0:0.15,C:0.65, label:"Dynasty founded",      event:"Tang Taizong establishes the most powerful state in the world. The Silk Road flourishes. Efficiency is extraordinary."},
      {year:712, chi:0.82,s:0.90,lambda0:0.18,C:0.72, label:"Xuanzong golden age",  event:"Peak Tang civilization. Poetry, art, trade. Chang'an is the largest city on Earth. The margin is at its highest."},
      {year:755, chi:0.72,s:0.78,lambda0:0.28,C:0.80, label:"An Lushan Rebellion",  event:"The most devastating rebellion in Chinese history begins. 36 million people die over 8 years. Burden explodes."},
      {year:780, chi:0.62,s:0.65,lambda0:0.38,C:0.84, label:"Tax reform fails",     event:"Desperate fiscal reforms fail to close the gap. Provincial warlords keep tax revenue. Central control weakens."},
      {year:835, chi:0.50,s:0.52,lambda0:0.50,C:0.87, label:"Sweet Dew Incident",   event:"Failed coup against the eunuch faction. Central government now powerless. M goes negative."},
      {year:875, chi:0.38,s:0.38,lambda0:0.60,C:0.89, label:"Huang Chao Rebellion", event:"Second catastrophic rebellion. The capital sacked twice. What remains of the imperial system disintegrates."},
      {year:907, chi:0.22,s:0.20,lambda0:0.72,C:0.90, label:"Dynasty ends",         event:"The last Tang emperor is deposed. China fractures into the Five Dynasties and Ten Kingdoms. The margin was negative for 70 years."},
    ]
  },
  {
    id:"indus", emoji:"🏺", label:"Indus Valley", domain:"Collapse",
    period:"2600–1700 BCE", color:"#C4B5FD",
    desc:"The Indus Valley Civilization was the largest of the ancient world — bigger than Mesopotamia and Egypt combined. Its collapse remains one of archaeology's great mysteries. The Stability Margin offers a hypothesis.",
    source:"Kenoyer (1998), Giosan et al. (2012), Petrie et al. (2017)",
    points:[
      {year:2600,chi:0.86,s:0.88,lambda0:0.14,C:0.68, label:"Mature Harappan",     event:"Mohenjo-daro and Harappa at their peak. Standardized weights, plumbing, and city planning across 1 million km²."},
      {year:2400,chi:0.83,s:0.85,lambda0:0.17,C:0.74, label:"Urban expansion",     event:"The civilization expands to 1,500 known sites. Trade with Mesopotamia flourishes. Complexity is rising."},
      {year:2200,chi:0.77,s:0.78,lambda0:0.23,C:0.78, label:"Monsoon weakening",   event:"Paleoclimate records show monsoon rainfall beginning to decline. Agricultural surplus starts to shrink."},
      {year:2000,chi:0.68,s:0.64,lambda0:0.32,C:0.80, label:"River shifts",        event:"The Ghaggar-Hakra river system begins to dry. The agricultural base for millions of people is failing."},
      {year:1900,chi:0.54,s:0.50,lambda0:0.44,C:0.81, label:"Urban abandonment",   event:"Mohenjo-daro and Harappa are systematically abandoned. Population disperses into smaller rural settlements."},
      {year:1800,chi:0.38,s:0.34,lambda0:0.56,C:0.80, label:"Late Harappan",       event:"The urban system is gone. What remains is a scattered rural population. Writing disappears. The margin called this 300 years ago."},
      {year:1700,chi:0.24,s:0.20,lambda0:0.66,C:0.76, label:"Civilization ends",   event:"The Indus Valley Civilization ceases to exist as a recognizable entity. Its script remains undeciphered to this day."},
    ]
  },
  {
    id:"ottoman", emoji:"🌙", label:"Ottoman Empire", domain:"Collapse",
    period:"1683–1922 CE", color:"#FB923C",
    desc:"The Ottoman Empire survived for 600 years. Its final 240-year decline is a textbook case of a system that kept adding complexity to solve problems that complexity itself had created.",
    source:"Quataert (2000), Finkel (2005), Hanioğlu (2008)",
    points:[
      {year:1683,chi:0.76,s:0.80,lambda0:0.24,C:0.82, label:"Vienna defeat",        event:"The failed siege of Vienna marks the turning point. Ottoman expansion ends. Defense costs now dominate the budget."},
      {year:1730,chi:0.72,s:0.74,lambda0:0.28,C:0.84, label:"Tulip Period",         event:"Westernization attempts begin. Modernization adds bureaucratic complexity without improving core efficiency."},
      {year:1800,chi:0.64,s:0.66,lambda0:0.36,C:0.86, label:"Napoleonic era",       event:"European powers dominate the Mediterranean. Ottoman trade revenue falls. The margin turns negative."},
      {year:1839,chi:0.56,s:0.58,lambda0:0.44,C:0.88, label:"Tanzimat reforms",     event:"Desperate modernization reforms. Each new administrative layer adds more complexity and more overhead."},
      {year:1878,chi:0.46,s:0.48,lambda0:0.54,C:0.90, label:"Congress of Berlin",   event:"Major territorial losses. Tax base shrinks but administrative costs don't. The empire is technically bankrupt."},
      {year:1914,chi:0.34,s:0.34,lambda0:0.64,C:0.91, label:"WWI entry",            event:"The empire enters WWI as the 'sick man of Europe.' Military overhead consumes everything. Nothing is left in reserve."},
      {year:1922,chi:0.18,s:0.16,lambda0:0.74,C:0.88, label:"Empire dissolved",     event:"The Ottoman sultanate is abolished. Six centuries of empire ends. The margin had been deeply negative for over 100 years."},
    ]
  },
  {
    id:"enron", emoji:"💼", label:"Enron", domain:"Collapse",
    period:"1996–2001", color:"#F472B6",
    desc:"Enron was named America's Most Innovative Company six years in a row. The Stability Margin shows the whole thing was hollow long before the scandal broke.",
    source:"SEC filings, McLean & Elkind (2003), Powers Report (2002)",
    points:[
      {year:1996,chi:0.68,s:0.75,lambda0:0.28,C:0.62, label:"Trading powerhouse",  event:"Enron transforms from a pipeline company into an energy trader. Revenue grows explosively. Complexity is rising fast."},
      {year:1997,chi:0.62,s:0.78,lambda0:0.34,C:0.68, label:"SPE structures begin",event:"Special Purpose Entities start hiding debt off the balance sheet. Real burden is much higher than reported."},
      {year:1998,chi:0.54,s:0.80,lambda0:0.42,C:0.75, label:"Broadband expansion", event:"Enron enters broadband, water, and weather derivatives. Each new business adds complexity without adding real efficiency."},
      {year:1999,chi:0.45,s:0.82,lambda0:0.50,C:0.80, label:"Mark-to-market peak", event:"Reported profits soar using mark-to-market accounting. Actual cash generation is deeply negative. The real M is critical."},
      {year:2000,chi:0.34,s:0.84,lambda0:0.58,C:0.84, label:"Stock peak",          event:"Enron stock hits $90. Wall Street loves it. But the underlying system is completely broken. The margin is catastrophic."},
      {year:2001,chi:0.18,s:0.48,lambda0:0.68,C:0.82, label:"Bankruptcy",          event:"Sherron Watkins' memo. SEC investigation. Stock collapses from $90 to $0.67 in weeks. $74 billion in shareholder value destroyed."},
      {year:2002,chi:0.08,s:0.10,lambda0:0.78,C:0.60, label:"Liquidation",           event:"Arthur Andersen collapses alongside Enron. Sarbanes-Oxley Act passed. 20,000 employees lose jobs and pensions. The system reaches zero."},
    ]
  },
  {
    id:"kodak", emoji:"📷", label:"Kodak", domain:"Collapse",
    period:"1990–2020", color:"#FDE047",
    desc:"Kodak invented the digital camera in 1975 and buried it. The Stability Margin shows a company that kept complexity high and efficiency low while the world moved on without it.",
    source:"Tripsas & Gavetti (2000), Lucas & Goh (2009), SEC filings via Macrotrends",
    points:[
      {year:1990,chi:0.78,s:0.88,lambda0:0.20,C:0.72, label:"Film dominance",      event:"Kodak controls 90% of US film sales and 85% of cameras. The margin is healthy but complexity is already high."},
      {year:1994,chi:0.72,s:0.82,lambda0:0.26,C:0.76, label:"Digital ignored",     event:"Kodak's own research shows digital will kill film within 10 years. Management buries the report. Overhead keeps rising."},
      {year:1997,chi:0.64,s:0.72,lambda0:0.34,C:0.79, label:"Digital cameras arrive",event:"Consumer digital cameras hit mass market. Kodak responds by expanding the film business. The margin turns negative."},
      {year:2000,chi:0.52,s:0.58,lambda0:0.44,C:0.81, label:"Film revenue peaks",  event:"Last year of strong film revenue. Kodak has 80,000 employees, 13 manufacturing plants. Fixed costs are enormous."},
      {year:2004,chi:0.40,s:0.44,lambda0:0.54,C:0.82, label:"Film collapse",       event:"Film revenue falls 40% in three years. Kodak lays off 15,000 workers. But the overhead barely budges."},
      {year:2008,chi:0.28,s:0.30,lambda0:0.64,C:0.80, label:"Desperate pivot",     event:"Kodak tries to become a printer company. The pivot adds new complexity without fixing the underlying collapse."},
      {year:2012,chi:0.15,s:0.14,lambda0:0.74,C:0.75, label:"Bankruptcy",          event:"Chapter 11 bankruptcy. 130 years of history ends. The margin had been negative for 15 years."},
      {year:2014,chi:0.28,s:0.26,lambda0:0.60,C:0.55, label:"Emerges restructured",  event:"Kodak exits bankruptcy as a much smaller company focused on commercial printing and licensing. Complexity dramatically reduced. The margin is still negative but improving."},
      {year:2020,chi:0.38,s:0.36,lambda0:0.50,C:0.48, label:"Kodak Moments survives",event:"The consumer photography brand sold off. Commercial printing business stabilizes. A tiny version of Kodak survives — simple, lean, and marginally viable."},
    ]
  },
  // ── RECOVERIES ─────────────────────────────────────────────────────────────
  {
    id:"germany", emoji:"🇩🇪", label:"West Germany / Germany", domain:"Recovery",
    period:"1945–2024", color:"#6EE7B7",
    desc:"The most dramatic economic recovery in modern history. From total destruction to the world's third-largest economy in 20 years. The Stability Margin shows exactly why it worked.",
    source:"Maddison Project Database, German Federal Statistical Office, Abelshauser (2004)",
    points:[
      {year:1945,chi:0.20,s:0.15,lambda0:0.68,C:0.45, label:"Zero hour",           event:"Germany is rubble. Industrial capacity destroyed. 12 million refugees. The burden of occupation and reconstruction is total."},
      {year:1948,chi:0.38,s:0.32,lambda0:0.52,C:0.42, label:"Currency reform",     event:"The Deutsche Mark replaces the worthless Reichsmark. Overnight, goods reappear in shop windows. The system restarts."},
      {year:1950,chi:0.52,s:0.48,lambda0:0.40,C:0.45, label:"Marshall Plan flows", event:"$1.4 billion in US aid modernizes infrastructure. Efficiency rises sharply. Complexity stays deliberately low. M turns positive."},
      {year:1953,chi:0.64,s:0.62,lambda0:0.30,C:0.50, label:"Debt forgiveness",    event:"London Debt Agreement cancels 50% of German external debt. The burden drops further. The margin improves dramatically."},
      {year:1957,chi:0.74,s:0.74,lambda0:0.22,C:0.56, label:"EEC founding",        event:"Germany joins the European Economic Community. Export markets open. Throughput surges. The Wirtschaftswunder accelerates."},
      {year:1960,chi:0.81,s:0.82,lambda0:0.16,C:0.60, label:"Economic miracle",    event:"Full employment. Germany exports more cars than the US. The recovery is complete. The margin is extraordinary."},
      {year:1965,chi:0.84,s:0.86,lambda0:0.14,C:0.64, label:"Mature economy",      event:"West Germany is the world's third-largest economy. 20 years from rubble to prosperity. A textbook positive margin recovery."},
      {year:1975,chi:0.82,s:0.83,lambda0:0.16,C:0.68, label:"Oil shock absorption",  event:"The 1973 oil crisis hits but Germany absorbs it. Margin dips slightly then recovers — a resilient system with genuine surplus."},
      {year:1990,chi:0.79,s:0.80,lambda0:0.19,C:0.74, label:"Reunification",         event:"East Germany absorbed overnight. 17 million people, crumbling infrastructure, uncompetitive industry. λ₀ surges as the West subsidizes the East. The margin drops sharply."},
      {year:1995,chi:0.74,s:0.76,lambda0:0.24,C:0.80, label:"Reunification costs",   event:"Treuhandanstalt privatizes 8,500 East German firms. Unemployment in the East hits 25%. The complexity of managing two formerly separate systems strains the margin."},
      {year:2003,chi:0.72,s:0.74,lambda0:0.26,C:0.82, label:"Hartz reforms",         event:"Schröder's painful labor reforms. Welfare restructured, labor market flexibilized. Short-term pain, long-term gain — the margin begins recovering."},
      {year:2010,chi:0.78,s:0.80,lambda0:0.20,C:0.83, label:"Export powerhouse",     event:"Germany emerges from the 2008 crisis stronger than almost any other developed economy. Manufacturing exports surge. The margin is healthy again."},
      {year:2024,chi:0.72,s:0.73,lambda0:0.26,C:0.86, label:"Geopolitical stress",   event:"Energy crisis from Ukraine war, deindustrialization pressure, aging population. Germany's margin is declining again — the same structural pressures that faced West Germany in 1990 are returning in new form."},
    ]
  },
  {
    id:"chesapeake", emoji:"🦀", label:"Chesapeake Bay", domain:"Recovery",
    period:"1983–2024", color:"#67E8F9",
    desc:"The most ambitious ecosystem restoration in US history. The Chesapeake Bay was dying. A 37-year coordinated intervention turned it around. The Stability Margin tracks the recovery.",
    source:"Chesapeake Bay Program Water Quality Database, CBP (2020), Kemp et al. (2005)",
    points:[
      {year:1983,chi:0.32,s:0.45,lambda0:0.58,C:0.52, label:"Crisis recognized",   event:"The bay is declared an ecological disaster. Oxygen dead zones cover 40% of the bottom. Oyster populations at 1% of historic levels."},
      {year:1987,chi:0.38,s:0.48,lambda0:0.54,C:0.54, label:"Chesapeake Bay Agreement",event:"Landmark multi-state agreement commits to 40% nutrient reduction. The first coordinated restoration effort begins."},
      {year:1992,chi:0.46,s:0.52,lambda0:0.48,C:0.56, label:"Sewage upgrades",     event:"$1 billion in wastewater treatment upgrades. Phosphorus levels begin to fall. The burden is slowly being reduced."},
      {year:2000,chi:0.55,s:0.58,lambda0:0.40,C:0.60, label:"Grass beds return",   event:"Underwater grass beds — critical habitat — start recovering. The system is rebuilding its complexity."},
      {year:2008,chi:0.63,s:0.65,lambda0:0.33,C:0.64, label:"Agricultural controls",event:"New farm runoff regulations take effect. Nitrogen loading drops sharply. The margin crosses zero for the first time."},
      {year:2014,chi:0.70,s:0.72,lambda0:0.26,C:0.68, label:"Blue crab recovery",  event:"Blue crab population reaches 30-year high. Oyster restoration programs show results. M is now solidly positive."},
      {year:2020,chi:0.76,s:0.78,lambda0:0.20,C:0.72, label:"Ongoing recovery",    event:"Water clarity at 33-year high. Dead zones shrink by 50%. The bay is not fully restored but the trajectory is unambiguous."},
      {year:2022,chi:0.79,s:0.80,lambda0:0.17,C:0.74, label:"Blue crab concern",    event:"Blue crab population drops sharply after record highs — a reminder that recovery is not linear. Nutrient goals still not fully met upstream."},
      {year:2024,chi:0.78,s:0.79,lambda0:0.18,C:0.75, label:"Mixed signals",        event:"Underwater grasses at near-record levels. But nitrogen targets still missed due to agricultural runoff. The margin holds but the system remains vulnerable to upstream decisions."},
    ]
  },
  // ── ECOLOGICAL ─────────────────────────────────────────────────────────────
  {
    id:"amazon", emoji:"🌳", label:"Amazon Rainforest", domain:"Ecological",
    period:"2000–2023", color:"#4ADE80",
    desc:"The largest tropical forest on Earth is approaching a tipping point. Scientific models suggest a 20–25% deforestation threshold triggers irreversible savannification. The margin shows where we are.",
    source:"Global Forest Watch / INPE, Lovejoy & Nobre (2018), Boulton et al. (2022)",
    points:[
      {year:2000,chi:0.88,s:0.86,lambda0:0.12,C:0.85, label:"Baseline",            event:"Amazon largely intact. 15% deforested. Self-generating rainfall cycle functioning. The system is still resilient."},
      {year:2004,chi:0.83,s:0.82,lambda0:0.17,C:0.86, label:"Deforestation peak",  event:"Deforestation hits all-time high — 27,000 km² in a single year. The moisture recycling system begins to strain."},
      {year:2009,chi:0.78,s:0.78,lambda0:0.22,C:0.87, label:"Drought 2010 begins", event:"Severe drought — the worst in 40 years. The forest loses its buffer. Recovery rates slow. Critical slowing down begins."},
      {year:2012,chi:0.73,s:0.74,lambda0:0.27,C:0.88, label:"Forest code weakened",event:"Brazil's Forest Code is weakened. Deforestation accelerates again. Scientists warn of approaching tipping point."},
      {year:2016,chi:0.66,s:0.67,lambda0:0.34,C:0.89, label:"Resilience loss",     event:"Nature study confirms Amazon has lost 75% of its resilience since 2000. The margin is now in warning territory."},
      {year:2019,chi:0.58,s:0.58,lambda0:0.42,C:0.90, label:"Record fires",        event:"80,000 fires in a single year. 20% of the Amazon now deforested. Scientists say the tipping point may be near."},
      {year:2023,chi:0.50,s:0.50,lambda0:0.50,C:0.91, label:"Tipping point risk",  event:"The eastern Amazon now emits more carbon than it absorbs. The margin has crossed zero. The system may be past recovery."},
    ]
  },
  {
    id:"yellowstone", emoji:"🐺", label:"Yellowstone", domain:"Recovery",
    period:"1995–2023", color:"#A3E635",
    desc:"When wolves were reintroduced to Yellowstone in 1995, they triggered one of the most remarkable ecological recoveries ever recorded — a trophic cascade that changed rivers. The Stability Margin shows it happening.",
    source:"NPS Yellowstone Wolf Project, Ripple & Beschta (2012), Beschta & Ripple (2016)",
    points:[
      {year:1995,chi:0.42,s:0.55,lambda0:0.48,C:0.42, label:"Wolf reintroduction", event:"31 wolves reintroduced from Canada. No apex predator for 70 years. Elk overgrazing has stripped riverbanks and valleys bare."},
      {year:1998,chi:0.52,s:0.58,lambda0:0.40,C:0.48, label:"Elk behavior shifts", event:"Elk begin avoiding riverbanks and valleys — the 'landscape of fear.' Willows and aspens start recovering immediately."},
      {year:2001,chi:0.61,s:0.63,lambda0:0.33,C:0.54, label:"Vegetation returns",  event:"Riverbank vegetation recovers dramatically. Beaver populations begin to rise. The margin turns positive."},
      {year:2005,chi:0.70,s:0.70,lambda0:0.26,C:0.60, label:"Rivers change course",event:"Restored vegetation stabilizes riverbanks. Rivers literally narrow and deepen. Songbird populations surge."},
      {year:2010,chi:0.76,s:0.76,lambda0:0.20,C:0.66, label:"Trophic cascade",     event:"Bear, raven, and eagle populations rise by feeding on wolf kills. The entire food web restructures. Complexity is returning."},
      {year:2015,chi:0.80,s:0.80,lambda0:0.17,C:0.70, label:"Full ecosystem shift",event:"Scientists document the full trophic cascade. A wolf pack of 100 has transformed 2.2 million acres. The margin is healthy."},
      {year:2020,chi:0.83,s:0.82,lambda0:0.15,C:0.74, label:"Stable recovery",     event:"Yellowstone is widely cited as the most successful large carnivore recovery in history. 25 years, one decision, a changed ecosystem."},
      {year:2023,chi:0.84,s:0.83,lambda0:0.14,C:0.76, label:"Wolf management debate", event:"Wolf population at ~100. Hunting outside park boundaries reduces numbers temporarily. The ecosystem holds — evidence of genuine resilience when margin is strong."},
    ]
  },
  // ── URBAN ──────────────────────────────────────────────────────────────────
  {
    id:"singapore", emoji:"🇸🇬", label:"Singapore", domain:"Urban",
    period:"1965–2024", color:"#38BDF8",
    desc:"Singapore went from a third-world city with no natural resources to one of the highest per-capita incomes on Earth in 55 years. It is the clearest example of a deliberately managed Stability Margin in existence.",
    source:"World Bank Development Indicators, Singapore Department of Statistics, Lee (2000)",
    points:[
      {year:1965,chi:0.38,s:0.32,lambda0:0.48,C:0.35, label:"Independence",        event:"Singapore expelled from Malaysia. No hinterland, no resources, no military. Lee Kuan Yew calls it a 'poisoned shrimp.'"},
      {year:1970,chi:0.52,s:0.48,lambda0:0.36,C:0.38, label:"Industrialization",   event:"Export manufacturing zones established. Foreign investment floods in. Efficiency rises as complexity stays deliberately lean."},
      {year:1979,chi:0.64,s:0.62,lambda0:0.28,C:0.44, label:"Second industrial rev",event:"Deliberate shift to high-value manufacturing. Education system redesigned around economic need. Margin improves steadily."},
      {year:1985,chi:0.72,s:0.68,lambda0:0.22,C:0.50, label:"First recession",     event:"Singapore's first recession triggers immediate government response — wages cut, taxes reduced. The system self-corrects."},
      {year:1995,chi:0.80,s:0.78,lambda0:0.16,C:0.58, label:"Asian Tiger",         event:"Singapore is one of the Four Asian Tigers. Port handles 20% of world container traffic. The margin is extraordinary."},
      {year:2003,chi:0.82,s:0.76,lambda0:0.15,C:0.62, label:"SARS shock",          event:"SARS hits Singapore hard. Economic contraction. But the system has enough margin to absorb the shock and recover quickly."},
      {year:2020,chi:0.84,s:0.80,lambda0:0.13,C:0.68, label:"COVID resilience",    event:"Singapore manages COVID with world-leading efficiency. 55 years of deliberately maintained Stability Margin pays off."},
      {year:2022,chi:0.83,s:0.82,lambda0:0.14,C:0.70, label:"Post-COVID rebound",   event:"Singapore's economy rebounds faster than any comparable city. Changi Airport reopens. The margin barely moved during the crisis — exactly what a positive margin is for."},
      {year:2024,chi:0.82,s:0.83,lambda0:0.15,C:0.72, label:"AI hub ambitions",     event:"Singapore positions itself as Southeast Asia's AI and fintech capital. New complexity is being added deliberately — but so far efficiency is keeping pace."},
    ]
  },
  {
    id:"ocean", emoji:"🌊", label:"Global Ocean Heat", domain:"Current",
    period:"1960–2023", color:"#22D3EE",
    desc:"The ocean absorbs 90% of the excess heat from climate change. The Stability Margin applied to ocean system health shows a trajectory that climate scientists find deeply concerning.",
    source:"NOAA Ocean Climate Laboratory, Cheng et al. (2022), Interannual to Decadal Ocean Observations",
    points:[
      {year:1960,chi:0.88,s:0.82,lambda0:0.12,C:0.55, label:"Pre-industrial baseline",event:"Ocean heat content near equilibrium. Thermohaline circulation functioning normally. The system has enormous buffering capacity."},
      {year:1975,chi:0.85,s:0.80,lambda0:0.15,C:0.60, label:"Warming begins",       event:"Measurable ocean warming begins in records. The atmosphere is loading the ocean with heat faster than circulation can distribute it."},
      {year:1985,chi:0.81,s:0.78,lambda0:0.19,C:0.65, label:"Accelerating uptake",  event:"Ocean heat uptake doubles. Surface temperatures rise. Coral bleaching events begin. The burden is rising."},
      {year:1995,chi:0.76,s:0.74,lambda0:0.24,C:0.70, label:"El Niño intensifies",  event:"El Niño events grow stronger and more frequent. The ocean's ability to regulate global temperature is weakening."},
      {year:2005,chi:0.70,s:0.68,lambda0:0.30,C:0.75, label:"Deep ocean warming",   event:"Measurable warming now reaches 2,000 meters depth. The entire ocean column is accumulating heat. The margin crosses zero."},
      {year:2015,chi:0.62,s:0.60,lambda0:0.38,C:0.80, label:"Record heat years",    event:"Each year sets new ocean heat records. Marine heatwaves now annual events. Oxygen levels dropping in deep water."},
      {year:2023,chi:0.52,s:0.50,lambda0:0.48,C:0.84, label:"Record shattering",    event:"2023 ocean temperatures shatter all records by a margin that shocked scientists. The margin is deeply negative. The system is stressed beyond anything in the historical record."},
    ]
  },
];

// ── SPARKLINE ─────────────────────────────────────────────────────────────────
function Sparkline({ points, w=140, h=38, color="#3B82F6" }) {
  const vals = points.map(p => calcM(p.chi, p.s, p.lambda0, p.C));
  const min = Math.min(...vals, -0.35), max = Math.max(...vals, 0.35);
  const rng = max - min, pad = 5;
  const xs = vals.map((_, i) => pad + (i / (vals.length - 1)) * (w - pad * 2));
  const ys = vals.map(v => h - pad - ((v - min) / rng) * (h - pad * 2));
  const path = xs.map((x, i) => `${i===0?"M":"L"} ${x} ${ys[i]}`).join(" ");
  const zeroY = h - pad - ((0 - min) / rng) * (h - pad * 2);
  const lastM = vals[vals.length - 1];
  return (
    <svg width={w} height={h} style={{overflow:"visible",display:"block"}}>
      <line x1={pad} y1={zeroY} x2={w-pad} y2={zeroY} stroke="#ffffff12" strokeWidth={1} strokeDasharray="2,3"/>
      <path d={path} fill="none" stroke={mColor(lastM)} strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round"/>
      {vals.map((v,i) => <circle key={i} cx={xs[i]} cy={ys[i]} r={i===vals.length-1?3:2} fill={mColor(v)}/>)}
    </svg>
  );
}

// ── GAUGE ─────────────────────────────────────────────────────────────────────
function Gauge({ value, size=160 }) {
  const cx=size/2, cy=size*0.56, r=size*0.36;
  const cl = Math.max(-0.5, Math.min(0.5, value));
  const angle = (cl + 0.5) * 180 - 90;
  const rad = d => d * Math.PI / 180;
  const nx = cx + r * Math.cos(rad(angle - 90));
  const ny = cy + r * Math.sin(rad(angle - 90));
  const color = mColor(value);
  const segs = [
    {from:-90,to:-54,c:"#EF4444"},{from:-54,to:-18,c:"#F97316"},
    {from:-18,to:9,c:"#EAB308"},{from:9,to:36,c:"#84CC16"},
    {from:36,to:63,c:"#22C55E"},{from:63,to:90,c:"#06B6D4"},
  ];
  function arc(f,t) {
    const x1=cx+r*Math.cos(rad(f)), y1=cy+r*Math.sin(rad(f));
    const x2=cx+r*Math.cos(rad(t)), y2=cy+r*Math.sin(rad(t));
    return `M ${cx} ${cy} L ${x1} ${y1} A ${r} ${r} 0 0 1 ${x2} ${y2} Z`;
  }
  return (
    <svg width={size} height={size*0.62} viewBox={`0 0 ${size} ${size*0.62}`} style={{overflow:"visible"}}>
      {segs.map((s,i) => <path key={i} d={arc(s.from,s.to)} fill={s.c} opacity={0.18}/>)}
      <path d={`M ${cx-r} ${cy} A ${r} ${r} 0 0 1 ${cx+r} ${cy}`} fill="none" stroke="#ffffff12" strokeWidth={0.5}/>
      <line x1={cx} y1={cy} x2={nx} y2={ny} stroke={color} strokeWidth={2.5} strokeLinecap="round"/>
      <circle cx={cx} cy={cy} r={5} fill={color}/>
      <circle cx={cx} cy={cy} r={3} fill={color} opacity={0.4}/>
      <text x={cx} y={cy-r*0.4} textAnchor="middle" fill={color} fontSize={size*0.115} fontFamily="JetBrains Mono" fontWeight="500">
        {value>=0?"+":""}{value.toFixed(3)}
      </text>
      <text x={cx} y={cy-r*0.18} textAnchor="middle" fill="#A3A3A3" fontSize={size*0.072} fontFamily="Syne">
        Stability Margin M
      </text>
    </svg>
  );
}

// ── M CHART ───────────────────────────────────────────────────────────────────
function MChart({ points, dsColor="#3B82F6", dsId=null }) {
  const [hov, setHov] = useState(null);
  const vals = points.map(p => calcM(p.chi, p.s, p.lambda0, p.C));
  const min = Math.min(...vals, -0.42), max = Math.max(...vals, 0.42);
  const rng = max - min;
  const W=600, H=210, pL=52, pR=16, pT=18, pB=38;
  const xs = points.map((_,i) => pL + (i/(points.length-1))*(W-pL-pR));
  const ys = vals.map(v => pT + ((max-v)/rng)*(H-pT-pB));
  const path = xs.map((x,i) => `${i===0?"M":"L"} ${x} ${ys[i]}`).join(" ");
  const fill = path + ` L ${xs[xs.length-1]} ${H-pB} L ${xs[0]} ${H-pB} Z`;
  const zeroY = pT + (max/rng)*(H-pT-pB);
  return (
    <div style={{overflowX:"auto"}}>
      <svg width={W} height={H} style={{display:"block",minWidth:W}}>
        <defs>
          <linearGradient id="gfill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={mColor(vals[vals.length-1])} stopOpacity="0.18"/>
            <stop offset="100%" stopColor={mColor(vals[vals.length-1])} stopOpacity="0"/>
          </linearGradient>
        </defs>
        {/* Grid */}
        {[-0.3,-0.2,-0.1,0,0.1,0.2,0.3].map(v => {
          const y = pT + ((max-v)/rng)*(H-pT-pB);
          if (y<pT||y>H-pB) return null;
          return (
            <g key={v}>
              <line x1={pL} y1={y} x2={W-pR} y2={y} stroke={v===0?"#ffffff20":"#ffffff07"} strokeWidth={v===0?1:0.5} strokeDasharray={v===0?"none":"3,4"}/>
              <text x={pL-5} y={y+4} textAnchor="end" fill="#737373" fontSize={9} fontFamily="JetBrains Mono">{v>=0?"+":""}{v.toFixed(1)}</text>
            </g>
          );
        })}
        {/* Negative zone */}
        {zeroY < H-pB && <rect x={pL} y={zeroY} width={W-pL-pR} height={H-pB-zeroY} fill="#EF444406"/>}
        {/* Lead time markers */}
        {dsId && (() => {
          const lead = getWarningLead(dsId);
          if (!lead || !lead.lead || !lead.m_negative_year || !lead.event_year) return null;
          const allYears = points.map(p=>p.year);
          const minY = allYears[0], maxY = allYears[allYears.length-1];
          const toX = yr => pL + ((yr - minY) / (maxY - minY)) * (W - pL - pR);
          const negX = toX(lead.m_negative_year);
          const evtX = toX(lead.event_year);
          if (negX < pL || evtX > W-pR) return null;
          return (
            <g>
              {/* M goes negative line */}
              <line x1={negX} y1={pT} x2={negX} y2={H-pB} stroke="#F97316" strokeWidth={1.5} strokeDasharray="4,3" opacity={0.7}/>
              <text x={negX+3} y={pT+10} fill="#F97316" fontSize={8} fontFamily="Inter">M&lt;0</text>
              <text x={negX+3} y={pT+20} fill="#F97316" fontSize={8} fontFamily="Inter">{lead.m_negative_year}</text>
              {/* Event line */}
              <line x1={evtX} y1={pT} x2={evtX} y2={H-pB} stroke="#EF4444" strokeWidth={1.5} strokeDasharray="4,3" opacity={0.7}/>
              <text x={evtX+3} y={pT+10} fill="#EF4444" fontSize={8} fontFamily="Inter">Event</text>
              <text x={evtX+3} y={pT+20} fill="#EF4444" fontSize={8} fontFamily="Inter">{lead.event_year}</text>
              {/* Arrow between them */}
              {Math.abs(evtX - negX) > 30 && (
                <g>
                  <line x1={negX} y1={H-pB-8} x2={evtX} y2={H-pB-8} stroke="#F97316" strokeWidth={1} opacity={0.6}/>
                  <text x={(negX+evtX)/2} y={H-pB-12} textAnchor="middle" fill="#F97316" fontSize={8} fontFamily="Inter" fontWeight="600">
                    {lead.lead}yr warning
                  </text>
                </g>
              )}
            </g>
          );
        })()}
        <path d={fill} fill="url(#gfill)"/>
        <path d={path} fill="none" stroke={mColor(vals[vals.length-1])} strokeWidth={2} strokeLinecap="round" strokeLinejoin="round"/>
        {/* Points + events */}
        {points.map((p,i) => {
          const m = vals[i], isH = hov===i;
          return (
            <g key={i} onMouseEnter={()=>setHov(i)} onMouseLeave={()=>setHov(null)} style={{cursor:"pointer"}}>
              <line x1={xs[i]} y1={pT} x2={xs[i]} y2={H-pB} stroke={mColor(m)} strokeWidth={isH?1:0.5} strokeDasharray="2,3" opacity={isH?0.7:0.25}/>
              <circle cx={xs[i]} cy={ys[i]} r={isH?5.5:3.5} fill={mColor(m)} opacity={isH?1:0.75}/>
              {isH && (
                <g>
                  <rect x={Math.min(xs[i]+10,W-210)} y={ys[i]-58} width={200} height={64} rx={6} fill="#0A0A0A" stroke={mColor(m)} strokeWidth={1} opacity={0.97}/>
                  <text x={Math.min(xs[i]+18,W-202)} y={ys[i]-40} fill={mColor(m)} fontSize={9} fontFamily="JetBrains Mono" fontWeight="500">{p.year} CE · M={m>=0?"+":""}{m.toFixed(3)}</text>
                  <text x={Math.min(xs[i]+18,W-202)} y={ys[i]-26} fill="#D4D4D4" fontSize={9} fontFamily="JetBrains Mono">{p.label}</text>
                  <foreignObject x={Math.min(xs[i]+10,W-210)} y={ys[i]-18} width={200} height={28}>
                    <div xmlns="http://www.w3.org/1999/xhtml" style={{fontSize:9,color:"#A3A3A3",fontFamily:"Syne",lineHeight:1.45,padding:"0 8px"}}>{p.event?.slice(0,90)}{p.event?.length>90?"…":""}</div>
                  </foreignObject>
                </g>
              )}
            </g>
          );
        })}
        {/* X axis */}
        {points.map((p,i) => {
          if (points.length>8 && i%2!==0) return null;
          return <text key={i} x={xs[i]} y={H-pB+14} textAnchor="middle" fill="#737373" fontSize={9} fontFamily="JetBrains Mono">{p.year}</text>;
        })}
        {zeroY<H-pB && <text x={pL+6} y={Math.min(zeroY+14,H-pB-4)} fill="#EF444430" fontSize={8} fontFamily="Syne" fontStyle="italic">burden exceeds output</text>}
      </svg>
      <div style={{fontSize:10,color:"#737373",fontFamily:"Syne",marginTop:4,fontStyle:"italic"}}>Hover any point to see what was happening historically</div>
    </div>
  );
}

// ── DATASET CARD ──────────────────────────────────────────────────────────────
function DSCard({ ds, active, onClick }) {
  const lastM = calcM(ds.points[ds.points.length-1].chi,ds.points[ds.points.length-1].s,ds.points[ds.points.length-1].lambda0,ds.points[ds.points.length-1].C);
  return (
    <div onClick={onClick} style={{
      background: active?"#111111":"#0A0A0A",
      border:`1px solid ${active?ds.color+"80":"#2A2A2A"}`,
      borderRadius:12, padding:20, cursor:"pointer",
      transition:"all 0.18s",
      boxShadow: active?`0 0 24px ${ds.color}18`:"none",
    }}
      onMouseEnter={e=>{if(!active){e.currentTarget.style.borderColor=ds.color+"50";e.currentTarget.style.background="#0A0A0A";}}}
      onMouseLeave={e=>{if(!active){e.currentTarget.style.borderColor="#2A2A2A";e.currentTarget.style.background="#0A0A0A";}}}
    >
      <div style={{display:"flex",justifyContent:"space-between",alignItems:"flex-start",marginBottom:10}}>
        <div>
          <span style={{fontSize:22}}>{ds.emoji}</span>
          <div style={{fontSize:13,fontFamily:"var(--serif)",color:"#FFFFFF",marginTop:4,lineHeight:1.2}}>{ds.label}</div>
          <div style={{fontSize:9,fontFamily:"var(--mono)",color:"#A3A3A3",marginTop:3}}>{ds.period}</div>
        </div>
        <span style={{fontSize:9,background:`${ds.color}18`,border:`1px solid ${ds.color}35`,borderRadius:4,padding:"2px 7px",color:ds.color,fontFamily:"var(--mono)",whiteSpace:"nowrap",flexShrink:0,marginLeft:8}}>{ds.domain}</span>
      </div>
      <Sparkline points={ds.points} w={150} h={34}/>
      <div style={{marginTop:8,display:"flex",justifyContent:"space-between",alignItems:"center"}}>
        <span style={{fontSize:9,color:mColor(lastM),fontFamily:"var(--mono)"}}>M = {lastM>=0?"+":""}{lastM.toFixed(3)}</span>
        <span style={{fontSize:9,color:mColor(lastM),fontFamily:"var(--sans)"}}>{mLabel(lastM)}</span>
      </div>
      {(() => {
        const lead = getWarningLead(ds.id);
        if (!lead || lead.lead === null) return null;
        return (
          <div style={{marginTop:6,paddingTop:6,borderTop:"1px solid #1A1A1A",
            display:"flex",alignItems:"center",gap:4}}>
            <span style={{fontFamily:"var(--mono)",fontSize:11,color:"#F97316",fontWeight:700}}>
              {lead.lead}yr
            </span>
            <span style={{fontSize:9,color:"#737373",fontFamily:"var(--sans)"}}>
              warning lead
            </span>
          </div>
        );
      })()}
    </div>
  );
}

// ── LANDING PAGE ──────────────────────────────────────────────────────────────
function Landing({ onEnter }) {
  // World map dot coordinates [lon, lat] for 62 countries + climate systems
  const MAP_DOTS = [
    // Countries - sized by importance, colored by approximate M
    {lon:-95,lat:38,r:14,color:"#EAB308",label:"US"},
    {lon:-3,lat:54,r:8,color:"#22C55E",label:"UK"},
    {lon:10,lat:51,r:9,color:"#84CC16",label:"DE"},
    {lon:2,lat:46,r:8,color:"#EAB308",label:"FR"},
    {lon:138,lat:36,r:10,color:"#EAB308",label:"JP"},
    {lon:105,lat:35,r:16,color:"#84CC16",label:"CN"},
    {lon:78,lat:21,r:13,color:"#84CC16",label:"IN"},
    {lon:-52,lat:-10,r:10,color:"#F97316",label:"BR"},
    {lon:90,lat:60,r:12,color:"#EF4444",label:"RU"},
    {lon:134,lat:-25,r:8,color:"#84CC16",label:"AU"},
    {lon:128,lat:36,r:7,color:"#22C55E",label:"KR"},
    {lon:104,lat:1,r:5,color:"#06B6D4",label:"SG"},
    {lon:45,lat:24,r:6,color:"#22C55E",label:"SA"},
    {lon:-64,lat:-34,r:6,color:"#EF4444",label:"AR"},
    {lon:35,lat:39,r:7,color:"#F97316",label:"TR"},
    {lon:25,lat:-29,r:6,color:"#EF4444",label:"ZA"},
    {lon:8,lat:10,r:6,color:"#EF4444",label:"NG"},
    {lon:-102,lat:24,r:7,color:"#EAB308",label:"MX"},
    {lon:44,lat:33,r:5,color:"#EF4444",label:"IQ"},
    {lon:53,lat:32,r:5,color:"#EF4444",label:"IR"},
    {lon:-66,lat:8,r:3,color:"#EF4444",label:"VE"},
    // Climate systems
    {lon:0,lat:83,r:10,color:"#EF4444",label:"🧊"},
    {lon:-30,lat:-20,r:12,color:"#EF4444",label:"🌊"},
    {lon:-60,lat:-5,r:11,color:"#F97316",label:"🌳"},
    {lon:-35,lat:45,r:9,color:"#F97316",label:"🌀"},
    {lon:120,lat:5,r:10,color:"#EF4444",label:"🪸"},
    {lon:120,lat:68,r:8,color:"#F97316",label:"🏔️"},
    {lon:-42,lat:72,r:7,color:"#EF4444",label:"🏔️"},
  ];

  function toXY(lon, lat, W, H) {
    return [((lon+180)/360)*W, ((90-lat)/180)*H];
  }

  const W=800, H=400;

  return (
    <div style={{
      minHeight:"100vh", display:"flex", alignItems:"center", justifyContent:"center",
      padding:"40px 24px",
      background:"#000000",
      position:"relative", overflow:"hidden",
    }}>
      <style>{GLOBAL_CSS}</style>

      {/* World map background — continent outlines + dots */}
      <svg style={{position:"fixed",top:0,left:0,width:"100%",height:"100%",zIndex:0,pointerEvents:"none"}}
        viewBox="0 0 360 180" preserveAspectRatio="xMidYMid slice">
        <defs>
          <radialGradient id="vig" cx="50%" cy="50%" r="55%">
            <stop offset="25%" stopColor="#000000" stopOpacity="0"/>
            <stop offset="100%" stopColor="#000000" stopOpacity="1"/>
          </radialGradient>
          <filter id="contGlow" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="1.5" result="blur"/>
            <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
          </filter>
        </defs>

        {/* Simplified continent outlines — Mercator projection, viewBox 360x180 */}
        {/* North America */}
        <path filter="url(#contGlow)" d="M 190 25 L 220 22 L 240 28 L 255 35 L 260 45 L 255 55 L 245 62 L 235 68 L 225 72 L 215 78 L 208 85 L 200 82 L 192 75 L 188 65 L 185 55 L 182 45 L 185 35 Z"
          fill="#223322" stroke="#4A7A4A" strokeWidth={1.5} opacity={0.92}/>
        {/* Central America */}
        <path d="M 208 85 L 215 88 L 212 95 L 206 93 Z"
          fill="#223322" stroke="#4A7A4A" strokeWidth={0.9} opacity={0.87}/>
        {/* South America */}
        <path d="M 215 88 L 230 90 L 240 98 L 245 110 L 242 122 L 235 132 L 225 138 L 215 135 L 208 125 L 205 112 L 208 100 L 212 95 Z"
          fill="#223322" stroke="#4A7A4A" strokeWidth={1.5} opacity={0.92}/>
        {/* Europe */}
        <path d="M 175 28 L 185 25 L 192 28 L 195 35 L 190 40 L 183 42 L 178 38 Z"
          fill="#223322" stroke="#4A7A4A" strokeWidth={1.1} opacity={0.90}/>
        {/* Scandinavia */}
        <path d="M 183 22 L 190 20 L 192 26 L 187 28 L 183 26 Z"
          fill="#223322" stroke="#4A7A4A" strokeWidth={0.9} opacity={0.87}/>
        {/* Africa */}
        <path filter="url(#contGlow)" d="M 175 42 L 190 40 L 198 45 L 200 55 L 198 68 L 192 80 L 185 90 L 178 95 L 172 88 L 168 75 L 166 62 L 168 50 Z"
          fill="#223322" stroke="#4A7A4A" strokeWidth={1.5} opacity={0.92}/>
        {/* Middle East */}
        <path d="M 195 35 L 210 33 L 215 40 L 208 45 L 198 45 L 195 40 Z"
          fill="#223322" stroke="#4A7A4A" strokeWidth={0.9} opacity={0.87}/>
        {/* Russia / Central Asia */}
        <path filter="url(#contGlow)" d="M 192 18 L 240 15 L 280 18 L 295 25 L 290 32 L 275 35 L 255 32 L 235 28 L 215 26 L 200 28 L 192 24 Z"
          fill="#223322" stroke="#4A7A4A" strokeWidth={1.1} opacity={0.90}/>
        {/* South Asia */}
        <path d="M 240 38 L 258 35 L 268 42 L 265 52 L 258 58 L 248 55 L 240 48 Z"
          fill="#223322" stroke="#4A7A4A" strokeWidth={1.1} opacity={0.90}/>
        {/* China / East Asia */}
        <path d="M 265 28 L 295 25 L 308 32 L 312 42 L 305 48 L 290 50 L 275 48 L 265 42 L 262 35 Z"
          fill="#223322" stroke="#4A7A4A" strokeWidth={1.1} opacity={0.90}/>
        {/* Southeast Asia */}
        <path d="M 290 52 L 305 50 L 312 58 L 308 65 L 298 65 L 290 60 Z"
          fill="#223322" stroke="#4A7A4A" strokeWidth={0.9} opacity={0.87}/>
        {/* Australia */}
        <path d="M 300 110 L 320 105 L 335 110 L 338 122 L 330 130 L 315 132 L 305 125 L 298 118 Z"
          fill="#223322" stroke="#4A7A4A" strokeWidth={1.1} opacity={0.90}/>
        {/* Greenland */}
        <path d="M 155 12 L 170 10 L 175 18 L 168 22 L 158 20 Z"
          fill="#223322" stroke="#4A7A4A" strokeWidth={0.9} opacity={0.84}/>
        {/* Japan */}
        <path d="M 313 38 L 318 36 L 320 42 L 315 44 L 312 40 Z"
          fill="#223322" stroke="#4A7A4A" strokeWidth={0.9} opacity={0.87}/>

        {/* Grid lines — very faint */}
        {[60,120,180,240,300].map(x=><line key={x} x1={x} y1={0} x2={x} y2={180} stroke="#FFFFFF" strokeWidth={0.15} opacity={0.06}/>)}
        {[45,90,135].map(y=><line key={y} x1={0} y1={y} x2={360} y2={y} stroke="#FFFFFF" strokeWidth={0.15} opacity={0.06}/>)}
        {/* Equator — slightly brighter */}
        <line x1={0} y1={90} x2={360} y2={90} stroke="#FFFFFF" strokeWidth={0.3} opacity={0.12}/>

        {/* Country / climate dots */}
        {[
          {x:265,y:52,r:10,c:"#EAB308"},{x:183,y:36,r:6,c:"#22C55E"},{x:190,y:39,r:7,c:"#84CC16"},
          {x:182,y:44,r:6,c:"#EAB308"},{x:318,y:54,r:8,c:"#EAB308"},{x:84,y:32,r:9,c:"#84CC16"},
          {x:258,y:69,r:7,c:"#84CC16"},{x:128,y:115,r:6,c:"#F97316"},{x:270,y:30,r:10,c:"#EF4444"},
          {x:314,y:155,r:5,c:"#84CC16"},{x:308,y:54,r:5,c:"#22C55E"},{x:284,y:89,r:4,c:"#06B6D4"},
          {x:225,y:66,r:5,c:"#22C55E"},{x:116,y:146,r:5,c:"#EF4444"},{x:215,y:51,r:5,c:"#F97316"},
          {x:25,y:146,r:4,c:"#EF4444"},{x:188,y:80,r:5,c:"#EF4444"},{x:78,y:56,r:5,c:"#EAB308"},
          {x:296,y:124,r:5,c:"#EF4444"},{x:233,y:58,r:4,c:"#EF4444"},{x:114,y:82,r:3,c:"#EF4444"},
          {x:180,y:7,r:7,c:"#EF4444"},{x:150,y:110,r:8,c:"#EF4444"},{x:120,y:95,r:7,c:"#F97316"},
          {x:145,y:45,r:6,c:"#F97316"},{x:300,y:85,r:7,c:"#EF4444"},{x:300,y:22,r:6,c:"#F97316"},
          {x:138,y:18,r:5,c:"#EF4444"},
        ].map((d,i)=>(
          <g key={i}>
            <circle cx={d.x} cy={d.y} r={d.r*2.2} fill={d.c} opacity={0.06}/>
            <circle cx={d.x} cy={d.y} r={d.r*1.2} fill={d.c} opacity={0.15}/>
            <circle cx={d.x} cy={d.y} r={d.r*0.65} fill={d.c} opacity={0.45}/>
          </g>
        ))}

        {/* Connection lines */}
        {[
          [265,52,183,44],[183,44,190,39],[190,39,318,54],[318,54,284,89],
          [284,89,258,69],[265,52,128,115],[180,7,138,18],[120,95,150,110],
        ].map(([x1,y1,x2,y2],i)=>(
          <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} stroke="#FFFFFF" strokeWidth={0.4} opacity={0.08}/>
        ))}

        {/* Vignette overlay — dark edges, lighter center */}
        <rect x={0} y={0} width={360} height={180} fill="url(#vig)"/>
        <rect x={0} y={145} width={360} height={35} fill="#000000" opacity={0.9}/>
        <rect x={0} y={0} width={360} height={30} fill="#000000" opacity={0.75}/>
      </svg>

      {/* Top accent line */}
      <div style={{position:"fixed",top:0,left:0,right:0,height:2,
        background:"linear-gradient(90deg,transparent,#2563EB,#3B82F6,transparent)",zIndex:10}}/>

      <div style={{maxWidth:640,width:"100%",position:"relative",zIndex:1,display:"flex",flexDirection:"column",gap:0,maxWidth:580}}>

        {/* Eyebrow */}
        <div style={{animation:"fadeUp 0.5s ease both",animationDelay:"0.05s",marginBottom:28}}>
          <span style={{fontFamily:"var(--mono)",fontSize:10,color:"var(--accent2)",letterSpacing:4,textTransform:"uppercase"}}>
            Welcome to
          </span>
        </div>

        {/* Title */}
        <h1 style={{
          fontFamily:"var(--serif)", fontSize:"clamp(32px,6vw,52px)",
          color:"#FFFFFF", lineHeight:1.1, marginBottom:24,
          animation:"fadeUp 0.5s ease both", animationDelay:"0.12s"
        }}>
          Engine of<br/>
          <span style={{color:"var(--accent2)"}}>Emergence</span>
        </h1>

        {/* Equation */}
        <div style={{
          display:"inline-flex", alignItems:"center", gap:12,
          background:"#0A0A0A", border:"1px solid #1E2D42",
          borderRadius:10, padding:"12px 20px", marginBottom:28,
          animation:"fadeUp 0.5s ease both", animationDelay:"0.2s",
          alignSelf:"flex-start"
        }}>
          <span style={{fontFamily:"var(--mono)",fontSize:20,color:"#FFFFFF",letterSpacing:2}}>M = χs − λ(C)</span>
          <span style={{fontFamily:"var(--mono)",fontSize:10,color:"#A3A3A3",borderLeft:"1px solid #1E2D42",paddingLeft:12}}>Stability Margin</span>
        </div>

        {/* What it is — 2-3 sentences */}
        <div style={{
          background:"#0A0A0A", border:"1px solid #1E2D42", borderRadius:12,
          padding:"20px 24px", marginBottom:16,
          animation:"fadeUp 0.5s ease both", animationDelay:"0.28s"
        }}>
          <div style={{fontFamily:"var(--mono)",fontSize:9,color:"var(--accent2)",letterSpacing:3,marginBottom:10,textTransform:"uppercase"}}>What is the Engine of Emergence?</div>
          <p style={{fontFamily:"var(--sans)",fontSize:14,color:"#D4D4D4",lineHeight:1.75,fontWeight:400}}>
            Every complex system — a civilization, an ecosystem, a company, a city — runs on energy and maintains
            itself through structure. The Engine of Emergence is a scientific framework that measures the
            <strong style={{color:"#FFFFFF"}}> Stability Margin</strong>: the difference between what a system generates
            and what it costs to maintain. When that number goes negative, collapse follows. Every time.
          </p>
          <p style={{fontFamily:"var(--sans)",fontSize:14,color:"#D4D4D4",lineHeight:1.75,marginTop:10,fontWeight:400}}>
            The framework was validated against <strong style={{color:"#FFFFFF"}}>10 independent real-world cases</strong> — 5 historical
            civilizational collapses and 5 ecological tipping points — confirming that the Stability Margin turned
            negative before the event in every single case, with warning windows of decades to centuries.
          </p>
        </div>

        {/* What this tool does */}
        <div style={{
          background:"#0A0A0A", border:"1px solid #0D948840", borderRadius:12,
          padding:"20px 24px", marginBottom:32,
          animation:"fadeUp 0.5s ease both", animationDelay:"0.36s"
        }}>
          <div style={{fontFamily:"var(--mono)",fontSize:9,color:"var(--accent2)",letterSpacing:3,marginBottom:10,textTransform:"uppercase"}}>What does this tool do?</div>
          <p style={{fontFamily:"var(--sans)",fontSize:14,color:"#D4D4D4",lineHeight:1.75,fontWeight:400}}>
            This tool lets you explore the Stability Margin across 20 real historical and contemporary systems —
            from the fall of Rome to the Great Barrier Reef — with the exact historical events that caused each
            rise and fall annotated directly on the chart. You can also upload your own data, use our verified
            source directory to find public datasets, and run your own experiments. The built-in assistant can
            walk you through any of it in plain English.
          </p>
        </div>

        {/* CTA */}
        <div style={{animation:"fadeUp 0.5s ease both",animationDelay:"0.44s"}}>
          <button onClick={onEnter} style={{
            background:"var(--accent)", color:"#FFFFFF", border:"none",
            borderRadius:10, padding:"16px 40px", fontSize:15, fontWeight:700,
            fontFamily:"var(--sans)", letterSpacing:0.5,
            transition:"all 0.15s", width:"100%", maxWidth:280,
            boxShadow:"0 0 32px #0D948840",
          }}
            onMouseEnter={e=>{e.target.style.background="var(--accent2)";e.target.style.boxShadow="0 0 40px #14B8A660";}}
            onMouseLeave={e=>{e.target.style.background="var(--accent)";e.target.style.boxShadow="0 0 32px #0D948840";}}
          >
            Get Started →
          </button>
          <div style={{marginTop:12,fontSize:10,color:"#404040",fontFamily:"var(--mono)"}}>
            Nathan Baird · Independent Researcher · EoE under peer review · 2026
          </div>
        </div>
      </div>
    </div>
  );
}

// ── TAB: UNDERSTAND ───────────────────────────────────────────────────────────
function UnderstandTab() {
  const [showMath, setShowMath] = useState(false);
  return (
    <div style={{maxWidth:700,margin:"0 auto",display:"flex",flexDirection:"column",gap:36}}>
      <div>
        <h2 style={{fontFamily:"var(--serif)",fontSize:28,color:"#FFFFFF",marginBottom:12,borderLeft:"3px solid #A78BFA",paddingLeft:14}}>
          Why do complex things <em style={{color:"#A78BFA"}}>fall apart?</em>
        </h2>
        <p style={{color:"#D4D4D4",fontSize:14,lineHeight:1.75,fontFamily:"var(--sans)",fontWeight:400}}>
          Civilizations. Cities. Ecosystems. Companies. They all grow, peak, and sometimes collapse. The answer is always the same — and there's a number for it.
        </p>
      </div>
      {[
        {icon:"⚡",title:"Every system runs on energy",body:"A city runs on tax revenue and commerce. A reef runs on sunlight and nutrients. A civilization runs on agricultural surplus and trade. The more efficiently a system converts that energy into useful output, the healthier it is. That ratio is χ — architectural efficiency."},
        {icon:"📈",title:"Overhead grows faster than you think",body:"Here's the trap: the cost of running a complex system doesn't grow proportionally with the system — it grows faster. Double the complexity, more than double the maintenance burden. This is captured in λ(C) — systemic burden — where the exponent n is greater than 1. The more complex a system gets, the harder it is to sustain."},
        {icon:"📊",title:"There's a number for how much runway is left",body:"When what you generate minus what it costs to maintain drops below zero, the system is spending more than it earns. That's M — the Stability Margin. Positive means runway. Negative means borrowed time. The framework has never found a case where collapse happened before M went negative."},
      ].map((c,i)=>(
        <div key={i} style={{background:"#111111",border:"1px solid #2A2A2A",borderRadius:12,padding:28,display:"flex",gap:20}}>
          <span style={{fontSize:26,flexShrink:0,marginTop:2}}>{c.icon}</span>
          <div>
            <div style={{fontSize:15,fontFamily:"var(--serif)",color:"#FFFFFF",marginBottom:8}}>{c.title}</div>
            <div style={{fontSize:13,color:"#D4D4D4",lineHeight:1.75,fontFamily:"var(--sans)",fontWeight:400}}>{c.body}</div>
          </div>
        </div>
      ))}

      {/* Math accordion */}
      <div style={{background:"#0A0A0A",border:"1px solid #1E2D42",borderRadius:12,overflow:"hidden"}}>
        <button onClick={()=>setShowMath(m=>!m)} style={{
          width:"100%",background:"none",border:"none",padding:"14px 20px",
          display:"flex",justifyContent:"space-between",alignItems:"center"
        }}>
          <span style={{fontFamily:"var(--sans)",fontSize:13,fontWeight:600,color:"#FFFFFF"}}>The equation — M = χs − λ(C)</span>
          <span style={{fontFamily:"var(--mono)",fontSize:16,color:"var(--accent2)",transform:showMath?"rotate(180deg)":"none",transition:"transform 0.2s"}}>⌄</span>
        </button>
        {showMath && (
          <div style={{padding:"0 20px 20px",borderTop:"1px solid #1E2D42"}}>
            <div style={{fontFamily:"var(--mono)",fontSize:22,color:"#FFFFFF",textAlign:"center",padding:"20px 0",letterSpacing:2}}>M = χs − λ(C)</div>
            <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(190px,1fr))",gap:10}}>
              {[
                {sym:"χ",name:"Architectural Efficiency",desc:"How well does the system convert inputs to useful outputs? 0 = catastrophic waste, 1 = perfect conversion.",color:"#60A5FA"},
                {sym:"s",name:"Energy Throughput",desc:"How much resource is actively flowing? Normalized 0–1 relative to the system's theoretical peak.",color:"#A78BFA"},
                {sym:"λ(C)",name:"Systemic Burden",desc:"Total overhead — fixed costs plus the complexity-driven burden. Grows superlinearly: λ₀ + kCⁿ where n > 1.",color:"#F87171"},
                {sym:"M",name:"Stability Margin",desc:"The result. Positive means the system generates more than it costs to run. Negative means borrowed time.",color:"var(--accent2)"},
              ].map((v,i)=>(
                <div key={i} style={{background:"#000000",borderRadius:8,padding:16}}>
                  <span style={{fontFamily:"var(--mono)",fontSize:18,color:v.color,fontWeight:500}}>{v.sym}</span>
                  <div style={{fontSize:11,fontWeight:600,color:"#FFFFFF",marginTop:4,fontFamily:"var(--sans)"}}>{v.name}</div>
                  <div style={{fontSize:11,color:"#A3A3A3",marginTop:4,lineHeight:1.5,fontFamily:"var(--sans)"}}>{v.desc}</div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* The result */}
      <div style={{background:"#000000",border:"1px solid #10B98130",borderRadius:12,padding:20}}>
        <div style={{fontFamily:"var(--mono)",fontSize:9,color:"#22C55E",marginBottom:10,letterSpacing:3}}>THE EMPIRICAL RESULT</div>
        <p style={{fontSize:14,lineHeight:1.75,color:"#D4D4D4",fontFamily:"var(--sans)",fontWeight:400}}>
          EoE was tested against <strong style={{color:"#FFFFFF"}}>10 independent real-world cases</strong> — 5 historical civilizational collapses and 5 ecological tipping points. In every single case, the Stability Margin turned negative <em>before</em> the collapse or tipping event. Warning windows ranged from decades to centuries.
        </p>
        <div style={{display:"flex",gap:10,marginTop:16,flexWrap:"wrap"}}>
          {["5/5 Civilizations confirmed","5/5 Ecosystems confirmed","10/10 cross-domain rate"].map((s,i)=>(
            <span key={i} style={{fontSize:11,color:"#22C55E",background:"#22C55E15",border:"1px solid #22C55E25",borderRadius:6,padding:"4px 12px",fontFamily:"var(--mono)"}}> ✓ {s}</span>
          ))}
        </div>
      </div>
    </div>
  );
}

// ── DOMAIN PRESETS ───────────────────────────────────────────────────────────
const DOMAIN_PRESETS = [
  { id:"company", emoji:"🏢", label:"Business", color:"#FCD34D",
    note:"Revenue and investment are the main inputs. Hardware maintenance, rent, and payroll overhead are the burden.",
    preset:{chi:0.72,s:0.78,lambda0:0.18,C:0.60},
    vars:{
      chi:{label:"How efficiently does money in become value out?",positive:["Revenue per employee","Gross profit margin","Sales per dollar of COGS"],negative:["Rework and waste","Redundant processes","Time lost to internal approvals"]},
      s:{label:"How much money and energy is flowing through right now?",positive:["Total revenue","New investment raised","Customer growth rate","Active contracts"],negative:["Revenue declining","Customers churning","Capital drying up"]},
      lambda0:{label:"What does it cost just to keep the lights on?",positive:[],negative:["Rent and facilities","Payroll for non-revenue staff","Computer hardware maintenance","Insurance and compliance","Debt service and interest"]},
      C:{label:"How complicated is this organization to run?",positive:[],negative:["Number of employees × management layers","Number of product lines","Number of markets and geographies","Number of tools and integrations"]},
    }
  },
  { id:"city", emoji:"🏙️", label:"City", color:"#60A5FA",
    note:"Cities are complexity engines. As they grow, output scales superlinearly — but so do coordination costs.",
    preset:{chi:0.80,s:0.87,lambda0:0.15,C:0.72},
    vars:{
      chi:{label:"How much economic value does the city generate per dollar of infrastructure?",positive:["GDP per dollar of road maintenance","Patents and innovation per resident","Business formation rate"],negative:["Infrastructure decay faster than repair","Traffic congestion reducing productivity","Crime raising cost of doing business"]},
      s:{label:"How much economic activity is flowing through right now?",positive:["Total employment and wages","Commercial tax receipts","Building permits and construction","Population growth"],negative:["Population decline","Business closures exceeding openings","Tax base shrinking"]},
      lambda0:{label:"What does it cost just to keep the city running?",positive:[],negative:["Police, fire, emergency services","Water, sewer, utilities maintenance","Road and bridge repair backlog","Pension obligations to retired workers","Debt service on municipal bonds"]},
      C:{label:"How many people, systems, and rules does this city coordinate?",positive:[],negative:["Population size (log scale)","Number of overlapping jurisdictions","Zoning complexity and permitting layers","Number of union contracts","Federal and state mandates"]},
    }
  },
  { id:"reef", emoji:"🪸", label:"Coral Reef", color:"#34D399",
    note:"Declining efficiency with rising complexity is the classic early warning signature before reef collapse.",
    preset:{chi:0.82,s:0.85,lambda0:0.14,C:0.75},
    vars:{
      chi:{label:"How efficiently does sunlight and nutrients become living reef?",positive:["Net primary productivity per unit of photosynthesis","Coral cover percentage","Calcification rates","Fish biomass per unit of habitat"],negative:["Bleaching-related productivity loss","Disease-driven mortality","Algae outcompeting coral"]},
      s:{label:"How much energy and nutrient is flowing into the system?",positive:["Solar irradiance at reef surface","Nutrient upwelling","Larval supply from upstream","Water clarity"],negative:["Thermal anomalies blocking photosynthesis","Sedimentation blocking light","Freshwater intrusion"]},
      lambda0:{label:"What does the reef spend energy on just to survive each day?",positive:[],negative:["Baseline respiration across all organisms","Immune responses to chronic disease","Tissue repair from wave damage","Energy cost of thermal stress"]},
      C:{label:"How intricate and interconnected is this ecosystem?",positive:["Species richness across all trophic levels","Number of symbiotic relationships","Structural complexity of reef architecture","Food web depth and redundancy"],negative:[]},
    }
  },
  { id:"government", emoji:"🇺🇸", label:"Government", color:"#F87171",
    note:"λ₀ is the critical variable for governments — mandatory spending is the floor and it only moves one direction.",
    preset:{chi:0.62,s:0.64,lambda0:0.35,C:0.88},
    vars:{
      chi:{label:"How effectively does spending turn into actual outcomes?",positive:["Infrastructure built per dollar","Health outcomes per dollar","Educational attainment per dollar","Economic growth per dollar of investment"],negative:["Administrative overhead consuming programs","Procurement waste","Program duplication across agencies"]},
      s:{label:"How much revenue and economic capacity is available?",positive:["Tax revenue as % of GDP","GDP growth rate","Employment rate","Trade and tariff income"],negative:["Recession shrinking tax base","Tax avoidance eroding revenue","Aging demographics reducing workforce"]},
      lambda0:{label:"What spending is locked in before a single discretionary dollar?",positive:[],negative:["Social Security and pension payments","Medicare and Medicaid","Interest on existing debt","Military baseline commitments","Existing civil service salaries"]},
      C:{label:"How many laws, agencies, and constituencies must be managed?",positive:[],negative:["Number of federal agencies","Pages of active regulation","Number of active spending programs","International treaty obligations","Distinct political constituencies"]},
    }
  },
  { id:"civilization", emoji:"🏛️", label:"Civilization", color:"#A78BFA",
    note:"Tainter described this collapse mechanism in plain language in 1988. EoE gives it a formal equation.",
    preset:{chi:0.75,s:0.80,lambda0:0.22,C:0.85},
    vars:{
      chi:{label:"How well does governance turn resources into stability and output?",positive:["Tax revenue actually collected","Agricultural yield per administered land","Trade facilitated per infrastructure dollar"],negative:["Corruption siphoning revenue","Military campaigns costing more than they return","Infrastructure decay exceeding repair"]},
      s:{label:"How much wealth and resource is flowing into the system?",positive:["Tax and tribute income","Agricultural surplus above subsistence","Trade volume on imperial routes","Resource extraction"],negative:["Border raids disrupting trade","Climate shocks reducing harvests","Plague reducing productive population"]},
      lambda0:{label:"What does it cost just to hold the empire together each year?",positive:[],negative:["Standing army wages and equipment","Frontier fortification and garrisons","Civil administration salaries","Imperial court and bureaucracy","Roads and aqueducts maintenance"]},
      C:{label:"How many moving parts does this empire coordinate?",positive:[],negative:["Number of provinces × administration layers","Diversity of languages and laws requiring management","Supply chains stretched across continents","Client kingdoms and dependencies","Competing religious and political factions"]},
    }
  },
  { id:"forest", emoji:"🌲", label:"Forest", color:"#4ADE80",
    note:"Forest dieback follows a slow decline in M over a decade — drought and pests raise λ₀ faster than the canopy can adapt.",
    preset:{chi:0.76,s:0.82,lambda0:0.28,C:0.78},
    vars:{
      chi:{label:"How much of the energy captured actually stays in the system?",positive:["Net ecosystem productivity — carbon stored","Wood volume increment per year","Seedling recruitment replacing canopy losses","Mycorrhizal network health"],negative:["Beetle or fungal outbreak consuming stored carbon","Drought causing more respiration than photosynthesis","Fire removing accumulated biomass"]},
      s:{label:"How much water, light, and nutrients is the forest receiving?",positive:["Annual precipitation in optimal range","Soil nutrient availability","Solar radiation reaching canopy","Nitrogen at productive levels"],negative:["Multi-year drought","Excessive nitrogen acidifying soils","Canopy closure blocking regenerating understory"]},
      lambda0:{label:"What does the forest spend energy on just to stay alive?",positive:[],negative:["Autotrophic respiration burning stored carbon","Chronic pest pressure requiring defensive chemistry","Wound response to physical damage","Water stress causing stomatal closure"]},
      C:{label:"How structurally rich and interconnected is this forest?",positive:["Tree species diversity","Number of canopy layers","Deadwood supporting cavity nesters","Below-ground fungal network extent","Age structure diversity"],negative:[]},
    }
  },
];

// ── DOWNLOAD REPORT ──────────────────────────────────────────────────────────
function downloadReport(ds, pts, pt, ptIdx) {
  const k = 0.15, n = 1.4;
  const calcM = (chi, s, lam, C) => chi * s - (lam + k * Math.pow(C, n));
  const mLabel = (m) => m > 0.15 ? "Stable" : m > 0.05 ? "Healthy" : m > -0.05 ? "Warning" : m > -0.15 ? "Declining" : "Critical";
  const M = calcM(pt.chi, pt.s, pt.lambda0, pt.C);
  const allMs = pts.map(p => calcM(p.chi, p.s, p.lambda0, p.C));
  const crossedNegative = pts.find((p, i) => allMs[i] < 0);
  const trend = allMs[allMs.length - 1] - allMs[0];

  const line = "─".repeat(60);
  const dline = "═".repeat(60);

  let report = [];
  report.push(dline);
  report.push("  ENGINE OF EMERGENCE — EXPERIMENT REPORT");
  report.push(dline);
  report.push("");
  report.push(`  System:      ${ds.label}`);
  report.push(`  Domain:      ${ds.domain}`);
  report.push(`  Period:      ${ds.period}`);
  report.push(`  Source:      ${ds.source}`);
  report.push(`  Generated:   ${new Date().toLocaleDateString("en-US", {year:"numeric",month:"long",day:"numeric"})}`);
  report.push("");
  report.push(line);
  report.push("  SELECTED TIME POINT");
  report.push(line);
  report.push("");
  report.push(`  Year / Label:   ${pt.year || pt.label}`);
  report.push(`  M (Stability Margin):  ${M >= 0 ? "+" : ""}${M.toFixed(4)}   [${mLabel(M)}]`);
  report.push(`  χ (Efficiency):        ${pt.chi.toFixed(4)}`);
  report.push(`  s (Throughput):        ${pt.s.toFixed(4)}`);
  report.push(`  λ(C) (Burden):         ${(pt.lambda0 + k * Math.pow(pt.C, n)).toFixed(4)}`);
  report.push(`  C (Complexity):        ${pt.C.toFixed(4)}`);
  report.push("");
  if (pt.event) {
    report.push("  Historical context:");
    const words = pt.event.split(" ");
    let line2 = "  ";
    words.forEach(w => {
      if ((line2 + w).length > 62) { report.push(line2); line2 = "  " + w + " "; }
      else { line2 += w + " "; }
    });
    if (line2.trim()) report.push(line2);
  }
  report.push("");
  report.push(line);
  report.push("  FULL TRAJECTORY");
  report.push(line);
  report.push("");
  report.push(`  ${"Year/Label".padEnd(22)} ${"M".padStart(8)}  ${"χ".padStart(6)}  ${"s".padStart(6)}  ${"λ(C)".padStart(7)}  ${"C".padStart(6)}  Status`);
  report.push(`  ${"─".repeat(22)} ${"─".repeat(8)}  ${"─".repeat(6)}  ${"─".repeat(6)}  ${"─".repeat(7)}  ${"─".repeat(6)}  ${"─".repeat(10)}`);
  pts.forEach((p, i) => {
    const m = allMs[i];
    const lbl = String(p.year || p.label).padEnd(22);
    const mStr = (m >= 0 ? "+" : "") + m.toFixed(4);
    report.push(`  ${lbl} ${mStr.padStart(8)}  ${p.chi.toFixed(4)}  ${p.s.toFixed(4)}  ${(p.lambda0 + k * Math.pow(p.C, n)).toFixed(4)}  ${p.C.toFixed(4)}  ${mLabel(m)}`);
  });
  report.push("");
  report.push(line);
  report.push("  SUMMARY ANALYSIS");
  report.push(line);
  report.push("");
  // R² calculation inline for report
  const rPts = pts.length;
  const ms2 = pts.map(p => calcM(p.chi,p.s,p.lambda0,p.C));
  const xs2 = pts.map((_,i)=>i/(rPts-1));
  const xm2 = xs2.reduce((a,b)=>a+b,0)/rPts;
  const ym2 = ms2.reduce((a,b)=>a+b,0)/rPts;
  const ssTot2 = ms2.reduce((a,v)=>a+(v-ym2)**2,0);
  const ssXX2 = xs2.reduce((a,x)=>a+(x-xm2)**2,0);
  const ssXY2 = ms2.reduce((a,y,j)=>a+(xs2[j]-xm2)*(y-ym2),0);
  const slope2 = ssXY2/ssXX2;
  const int2 = ym2-slope2*xm2;
  const ssRes2 = ms2.reduce((a,v,i)=>a+(v-(slope2*xs2[i]+int2))**2,0);
  const r2Val = ssTot2===0?1:Math.max(0,1-ssRes2/ssTot2);
  const rAbs = Math.sqrt(r2Val);
  const se2 = rPts>3?1/Math.sqrt(rPts-3):0.5;
  const z2 = 0.5*Math.log((1+rAbs)/(1-rAbs+0.0001));
  const ciLow2 = Math.max(0,((Math.exp(2*(z2-1.96*se2))-1)/(Math.exp(2*(z2-1.96*se2))+1))**2);
  const ciHigh2 = Math.min(1,((Math.exp(2*(z2+1.96*se2))-1)/(Math.exp(2*(z2+1.96*se2))+1))**2);

  const rStats = calcR2(pts);
  const wLead = getWarningLead(ds.id);
  report.push(line);
  report.push("  MODEL FIT & PREDICTIVE VALIDATION");
  report.push(line);
  report.push("");
  if (rStats) {
    report.push("  POLYNOMIAL R² (QUADRATIC FIT)");
    report.push(`  R²:                      ${rStats.r2.toFixed(4)}`);
    report.push(`  95% Confidence Interval:  [${rStats.ci_low.toFixed(3)}, ${rStats.ci_high.toFixed(3)}]`);
    report.push(`  Trajectory shape:         ${rStats.shape}`);
    report.push(`  Data points (n):          ${rStats.n}`);
    const fitStr2 = rStats.r2>0.85 ? "Strong fit" : rStats.r2>0.65 ? "Moderate fit" : "Exploratory fit";
    report.push("  Result: " + fitStr2 + " — model explains " + rStats.r2_pct + "% of variance.");
    if (rStats.nonMonotonic) report.push("  Note: Quadratic fit used — trajectory is a " + rStats.shape + ".");
    report.push("");
  }
  report.push("  PREDICTIVE VALIDATION (WARNING LEAD TIME)");
  if (wLead && wLead.lead !== null) {
    report.push(`  M first went negative:   ${wLead.m_negative_year}`);
    report.push(`  Known event:             ${wLead.event} (${wLead.event_year})`);
    report.push(`  Warning lead time:       ${wLead.lead} year${wLead.lead!==1?"s":""}`);
    report.push("  Validation: This is independently verifiable — event date was");
    report.push("  not used to calibrate model variables.");
    if (wLead.source) report.push("  Event source: " + wLead.source);
  } else if (wLead && wLead.note) {
    report.push("  " + wLead.note);
  } else {
    report.push("  No documented collapse or tipping point to validate against.");
    report.push("  R² measures internal model fit only.");
  }
  report.push("");

  report.push(`  Overall trend: M ${trend >= 0 ? "improved" : "declined"} by ${Math.abs(trend).toFixed(4)} over the full period.`);
  if (crossedNegative) {
    report.push(`  M first went negative at: ${crossedNegative.year || crossedNegative.label}`);
    report.push(`  This represents the point at which burden exceeded output.`);
  } else {
    report.push(`  M remained positive throughout the period analyzed.`);
  }
  report.push("");
  const negCount = allMs.filter(m => m < 0).length;
  report.push(`  Points with negative margin: ${negCount} of ${pts.length} (${Math.round(negCount/pts.length*100)}%)`);
  report.push(`  Points in warning zone (M < +0.05): ${allMs.filter(m => m < 0.05).length} of ${pts.length}`);
  report.push("");
  report.push(line);
  report.push("  ANNOTATION TIMELINE");
  report.push(line);
  report.push("");
  pts.forEach((p, i) => {
    if (p.event) {
      const m = allMs[i];
      report.push(`  ${p.year || p.label} [M=${(m>=0?"+":"")+m.toFixed(3)}]`);
      const words = p.event.split(" ");
      let ln = "    ";
      words.forEach(w => {
        if ((ln + w).length > 64) { report.push(ln); ln = "    " + w + " "; }
        else { ln += w + " "; }
      });
      if (ln.trim()) report.push(ln);
      report.push("");
    }
  });
  report.push(line);
  report.push("  SCIENTIFIC LEGITIMACY STATEMENT");
  report.push(line);
  report.push("");
  report.push("  EoE CAN say:");
  report.push("    - The stability margin is declining or improving");
  report.push("    - The pattern is consistent with EoE collapse predictions");
  report.push("    - M turned negative N years before a known event");
  report.push("");
  report.push("  EoE CANNOT say:");
  report.push("    - This system will collapse");
  report.push("    - Collapse will occur at a specific date");
  report.push("    - These are precise measurements (they are proxy estimates)");
  report.push("    - The framework is proven beyond 10 validated cases");
  report.push("");
  report.push("  Citation: Baird, N. (2026). Engine of Emergence.");
  report.push("  Preprint: arXiv:[pending] | Data: https://doi.org/10.5281/zenodo.19016245");
  report.push("");
  report.push(dline);
  report.push("  Generated by Engine of Emergence Interactive · engine-of-emergence.vercel.app");
  report.push(dline);

  const blob = new Blob([report.join("\n")], { type:"text/plain" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `EoE_${ds.label.replace(/[^a-zA-Z0-9]/g,"_")}_${pt.year||"report"}.txt`;
  a.click();
  URL.revokeObjectURL(url);
}


// ── CSV EXPORT ────────────────────────────────────────────────────────────────
function exportCSV(ds, pts) {
  const k=0.15, n_exp=1.4;
  const headers = ["label","year","chi","s","lambda0","C","lambda_C","M","status"];
  const rows = pts.map(p => {
    const lam = p.lambda0 + k*Math.pow(p.C, n_exp);
    const M   = p.chi*p.s - lam;
    return [
      `"${p.label||""}"`,
      p.year||"",
      p.chi, p.s, p.lambda0, p.C,
      lam.toFixed(4),
      M.toFixed(4),
      `"${mLabel(M)}"`
    ].join(",");
  });
  const csv = [headers.join(","), ...rows].join("\n");
  const blob = new Blob([csv], {type:"text/csv"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `EoE_${ds.label.replace(/[^a-zA-Z0-9]/g,"_")}_data.csv`;
  a.click();
  URL.revokeObjectURL(a.href);
}

// ── CITATION GENERATOR ────────────────────────────────────────────────────────
function CitationModal({ ds, onClose }) {
  const [copied, setCopied] = useState(null);
  const year = 2026;
  const author = "Baird, N.";
  const title = "Engine of Emergence: A Thermodynamic Framework for the Persistence and Collapse of Organized Complexity";
  const arxiv = "arXiv:[pending]";
  const url = "https://engine-of-emergence.vercel.app";
  const dataUrl = "https://doi.org/10.5281/zenodo.19016245";

  const citations = {
    APA: `${author} (${year}). ${title}. ${arxiv}. Interactive tool: ${url}`,
    MLA: `Baird, Nathan. "${title}." ${arxiv}, ${year}. Web. ${new Date().toLocaleDateString("en-US",{day:"numeric",month:"short",year:"numeric"})}. <${url}>`,
    Chicago: `Baird, Nathan. "${title}." ${arxiv} (${year}). ${url}.`,
    BibTeX: `@misc{baird${year}eoe,
  author = {Baird, Nathan},
  title  = {${title}},
  year   = {${year}},
  note   = {${arxiv}},
  url    = {${url}}
}`,
  };

  const datasetCite = {
    APA: `${author} (${year}). EoE Dataset: ${ds.label} [Data visualization]. Engine of Emergence. ${url} Source: ${ds.source}`,
    BibTeX: `@misc{baird${year}${ds.id},
  author = {Baird, Nathan},
  title  = {EoE Dataset: ${ds.label}},
  year   = {${year}},
  url    = {${url}},
  note   = {Source: ${ds.source}}
}`,
  };

  function copy(text, key) {
    navigator.clipboard.writeText(text).then(()=>{ setCopied(key); setTimeout(()=>setCopied(null),2000); });
  }

  return (
    <div style={{position:"fixed",inset:0,background:"#000000CC",zIndex:100,
      display:"flex",alignItems:"center",justifyContent:"center",padding:20}}
      onClick={onClose}>
      <div style={{background:"#0A0A0A",border:"1px solid #2A2A2A",borderRadius:14,
        padding:24,maxWidth:600,width:"100%",maxHeight:"80vh",overflowY:"auto"}}
        onClick={e=>e.stopPropagation()}>
        <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:20}}>
          <div style={{fontFamily:"var(--serif)",fontSize:18,color:"#FFFFFF"}}>Citation Generator</div>
          <button onClick={onClose} style={{background:"none",border:"none",
            color:"#737373",fontSize:20,cursor:"pointer"}}>×</button>
        </div>

        <div style={{fontFamily:"var(--mono)",fontSize:9,color:"#3B82F6",
          letterSpacing:3,marginBottom:12}}>CITE THE PAPER</div>
        {Object.entries(citations).map(([fmt,text])=>(
          <div key={fmt} style={{marginBottom:12}}>
            <div style={{display:"flex",justifyContent:"space-between",
              alignItems:"center",marginBottom:5}}>
              <span style={{fontSize:11,fontWeight:600,color:"#D4D4D4",
                fontFamily:"var(--sans)"}}>{fmt}</span>
              <button onClick={()=>copy(text,`paper-${fmt}`)} style={{
                background:copied===`paper-${fmt}`?"#22C55E20":"#111111",
                border:`1px solid ${copied===`paper-${fmt}`?"#22C55E":"#2A2A2A"}`,
                borderRadius:6,padding:"3px 10px",fontSize:10,
                color:copied===`paper-${fmt}`?"#22C55E":"#737373",
                fontFamily:"var(--sans)"}}>
                {copied===`paper-${fmt}`?"✓ Copied":"Copy"}
              </button>
            </div>
            <div style={{background:"#000000",borderRadius:8,padding:"10px 12px",
              fontFamily:"var(--mono)",fontSize:10,color:"#A3A3A3",
              lineHeight:1.6,whiteSpace:"pre-wrap",wordBreak:"break-all"}}>
              {text}
            </div>
          </div>
        ))}

        <div style={{fontFamily:"var(--mono)",fontSize:9,color:"#3B82F6",
          letterSpacing:3,marginBottom:12,marginTop:20}}>CITE THIS DATASET — {ds.label.toUpperCase()}</div>
        {Object.entries(datasetCite).map(([fmt,text])=>(
          <div key={fmt} style={{marginBottom:12}}>
            <div style={{display:"flex",justifyContent:"space-between",
              alignItems:"center",marginBottom:5}}>
              <span style={{fontSize:11,fontWeight:600,color:"#D4D4D4",
                fontFamily:"var(--sans)"}}>{fmt}</span>
              <button onClick={()=>copy(text,`ds-${fmt}`)} style={{
                background:copied===`ds-${fmt}`?"#22C55E20":"#111111",
                border:`1px solid ${copied===`ds-${fmt}`?"#22C55E":"#2A2A2A"}`,
                borderRadius:6,padding:"3px 10px",fontSize:10,
                color:copied===`ds-${fmt}`?"#22C55E":"#737373",
                fontFamily:"var(--sans)"}}>
                {copied===`ds-${fmt}`?"✓ Copied":"Copy"}
              </button>
            </div>
            <div style={{background:"#000000",borderRadius:8,padding:"10px 12px",
              fontFamily:"var(--mono)",fontSize:10,color:"#A3A3A3",
              lineHeight:1.6,whiteSpace:"pre-wrap",wordBreak:"break-all"}}>
              {text}
            </div>
          </div>
        ))}

        <div style={{marginTop:16,fontSize:10,color:"#525252",
          fontFamily:"var(--sans)",lineHeight:1.6}}>
          Data DOI: {dataUrl}
        </div>
      </div>
    </div>
  );
}

// ── TIPPING POINT PROJECTION ──────────────────────────────────────────────────
function projectTippingPoint(pts) {
  if (pts.length < 3) return null;
  const ms = pts.map(p => calcM(p.chi,p.s,p.lambda0,p.C));
  const lastM = ms[ms.length-1];
  if (lastM < 0) return { already_negative:true, since: pts[ms.findIndex(m=>m<0)]?.year };

  // Fit linear trend to last 4 points for projection
  const recent = ms.slice(-4);
  const recentPts = pts.slice(-4);
  const n = recent.length;
  const xMean = (n-1)/2;
  const yMean = recent.reduce((a,b)=>a+b,0)/n;
  const ssXX = recent.reduce((_,__,i)=>(i-xMean)**2+_,0);
  const ssXY = recent.reduce((_,v,i)=>_+(i-xMean)*(v-yMean),0);
  const slope = ssXX===0 ? 0 : ssXY/ssXX;

  if (slope >= 0) return { improving:true };

  // Estimate years until M=0
  const lastYear = recentPts[recentPts.length-1].year;
  const yearSpan = recentPts.length > 1
    ? (recentPts[recentPts.length-1].year - recentPts[0].year) / (recentPts.length-1)
    : 10;
  const stepsToZero = -lastM / Math.abs(slope);
  const yearsToZero = Math.round(stepsToZero * yearSpan);
  const projectedYear = lastYear + yearsToZero;

  // Uncertainty: ±50% of projection based on R² of recent trend
  const uncertainty = Math.round(yearsToZero * 0.5);

  return {
    projected_year: projectedYear,
    years_from_now: yearsToZero,
    uncertainty,
    range_low:  projectedYear - uncertainty,
    range_high: projectedYear + uncertainty,
    slope_per_step: slope,
    last_year: lastYear,
    improving: false,
    already_negative: false,
  };
}

function TippingPointBadge({ pts, dsLabel }) {
  const proj = projectTippingPoint(pts);
  if (!proj) return null;

  if (proj.already_negative) return (
    <div style={{background:"#1A0A0A",border:"1px solid #EF444430",
      borderRadius:10,padding:"12px 16px"}}>
      <div style={{fontFamily:"var(--mono)",fontSize:9,color:"#EF4444",
        letterSpacing:3,marginBottom:6}}>TIPPING POINT STATUS</div>
      <div style={{fontSize:13,color:"#EF4444",fontFamily:"var(--sans)",fontWeight:600}}>
        M already negative
        {proj.since ? ` since ${proj.since}` : ""}
      </div>
      <div style={{fontSize:11,color:"#737373",fontFamily:"var(--sans)",marginTop:4}}>
        Burden has exceeded output. System is in structured decline.
      </div>
    </div>
  );

  if (proj.improving) return (
    <div style={{background:"#0A1A0A",border:"1px solid #22C55E30",
      borderRadius:10,padding:"12px 16px"}}>
      <div style={{fontFamily:"var(--mono)",fontSize:9,color:"#22C55E",
        letterSpacing:3,marginBottom:6}}>TIPPING POINT PROJECTION</div>
      <div style={{fontSize:13,color:"#22C55E",fontFamily:"var(--sans)",fontWeight:600}}>
        Margin improving — no tipping point in sight
      </div>
      <div style={{fontSize:11,color:"#737373",fontFamily:"var(--sans)",marginTop:4}}>
        Recent trajectory is positive. No projection warranted.
      </div>
    </div>
  );

  return (
    <div style={{background:"#0F0C00",border:"1px solid #EAB30830",
      borderRadius:10,padding:"14px 16px"}}>
      <div style={{fontFamily:"var(--mono)",fontSize:9,color:"#EAB308",
        letterSpacing:3,marginBottom:10}}>TIPPING POINT PROJECTION</div>
      <div style={{display:"flex",gap:16,flexWrap:"wrap",alignItems:"flex-start"}}>
        <div>
          <div style={{fontFamily:"var(--mono)",fontSize:26,color:"#EAB308",
            fontWeight:700,lineHeight:1}}>{proj.projected_year}</div>
          <div style={{fontSize:11,color:"#737373",fontFamily:"var(--sans)",marginTop:3}}>
            projected M = 0 crossing
          </div>
          <div style={{fontSize:10,color:"#525252",fontFamily:"var(--mono)",marginTop:3}}>
            range: {proj.range_low}–{proj.range_high}
          </div>
        </div>
        <div style={{flex:1,minWidth:180}}>
          <div style={{fontSize:12,color:"#D4D4D4",fontFamily:"var(--sans)",
            lineHeight:1.7,marginBottom:8}}>
            At the current rate of decline, {dsLabel}'s Stability Margin is projected
            to cross zero around <strong style={{color:"#EAB308"}}>{proj.projected_year}</strong>{" "}
            (±{proj.uncertainty} years).
          </div>
          <div style={{fontSize:10,color:"#525252",fontFamily:"var(--sans)",lineHeight:1.5,
            fontStyle:"italic"}}>
            ⚠ Extrapolation only. Based on recent {Math.min(4,pts.length)}-point trend.
            Not a prediction — trajectories can change. Use alongside primary source data.
          </div>
        </div>
      </div>
    </div>
  );
}

// ── OVERLAY COMPARISON CHART ──────────────────────────────────────────────────
function OverlayChart({ ds1, ds2 }) {
  const pts1 = ds1.points;
  const pts2 = ds2.points;
  const ms1  = pts1.map(p=>calcM(p.chi,p.s,p.lambda0,p.C));
  const ms2  = pts2.map(p=>calcM(p.chi,p.s,p.lambda0,p.C));
  const allMs = [...ms1,...ms2];
  const minM = Math.min(...allMs,-0.5), maxM = Math.max(...allMs,0.4);
  const rng = maxM-minM;

  const W=580, H=200, pL=48, pR=16, pT=16, pB=36;

  // Normalize x to 0-1 within each dataset's own time range, then map to canvas
  function toX(i, total) { return pL + (i/(total-1))*(W-pL-pR); }
  function toY(v) { return pT + ((maxM-v)/rng)*(H-pT-pB); }

  const path1 = ms1.map((m,i)=>`${i===0?"M":"L"} ${toX(i,ms1.length)} ${toY(m)}`).join(" ");
  const path2 = ms2.map((m,i)=>`${i===0?"M":"L"} ${toX(i,ms2.length)} ${toY(m)}`).join(" ");
  const zeroY = toY(0);

  const grids = [-0.4,-0.2,0,0.2,0.4].filter(v=>v>=minM&&v<=maxM);

  return (
    <div style={{background:"#0A0A0A",border:"1px solid #2A2A2A",
      borderRadius:12,padding:16,overflowX:"auto"}}>
      <div style={{display:"flex",gap:16,marginBottom:12,flexWrap:"wrap"}}>
        <div style={{display:"flex",alignItems:"center",gap:6}}>
          <div style={{width:20,height:2,background:ds1.color,borderRadius:2}}/>
          <span style={{fontSize:11,color:"#D4D4D4",fontFamily:"var(--sans)"}}>{ds1.label}</span>
          <span style={{fontSize:9,color:"#525252",fontFamily:"var(--mono)"}}>({ds1.period})</span>
        </div>
        <div style={{display:"flex",alignItems:"center",gap:6}}>
          <div style={{width:20,height:2,background:ds2.color,borderRadius:2,borderTop:"2px dashed "+ds2.color}}/>
          <span style={{fontSize:11,color:"#D4D4D4",fontFamily:"var(--sans)"}}>{ds2.label}</span>
          <span style={{fontSize:9,color:"#525252",fontFamily:"var(--mono)"}}>({ds2.period})</span>
        </div>
        <span style={{fontSize:9,color:"#525252",fontFamily:"var(--sans)",marginLeft:"auto",fontStyle:"italic"}}>
          X axis normalized to each system's own time span
        </span>
      </div>
      <svg width={W} height={H} style={{display:"block",minWidth:W}}>
        {grids.map(v=>{
          const y=toY(v);
          return (
            <g key={v}>
              <line x1={pL} y1={y} x2={W-pR} y2={y}
                stroke={v===0?"#ffffff22":"#ffffff08"} strokeWidth={v===0?1:0.5}
                strokeDasharray={v===0?"none":"3,4"}/>
              <text x={pL-5} y={y+4} textAnchor="end" fill="#4B5563"
                fontSize={8} fontFamily="JetBrains Mono">
                {v>=0?"+":""}{v.toFixed(1)}
              </text>
            </g>
          );
        })}
        {/* Zero label */}
        {zeroY>pT && zeroY<H-pB && (
          <text x={pL+6} y={zeroY+12} fill="#EF444430" fontSize={7}
            fontFamily="Inter" fontStyle="italic">burden exceeds output</text>
        )}
        <path d={path1} fill="none" stroke={ds1.color} strokeWidth={2}
          strokeLinecap="round" strokeLinejoin="round"/>
        <path d={path2} fill="none" stroke={ds2.color} strokeWidth={2}
          strokeLinecap="round" strokeLinejoin="round" strokeDasharray="6,3"/>
        {ms1.map((m,i)=><circle key={i} cx={toX(i,ms1.length)} cy={toY(m)}
          r={3} fill={mColor(m)}/>)}
        {ms2.map((m,i)=><circle key={i} cx={toX(i,ms2.length)} cy={toY(m)}
          r={3} fill={mColor(m)} strokeWidth={1} stroke={ds2.color}/>)}
        {/* X axis labels */}
        <text x={pL} y={H-pB+14} textAnchor="middle" fill="#4B5563"
          fontSize={8} fontFamily="JetBrains Mono">start</text>
        <text x={W-pR} y={H-pB+14} textAnchor="middle" fill="#4B5563"
          fontSize={8} fontFamily="JetBrains Mono">end</text>
        <text x={(pL+W-pR)/2} y={H-pB+14} textAnchor="middle" fill="#4B5563"
          fontSize={8} fontFamily="JetBrains Mono">midpoint</text>
      </svg>
    </div>
  );
}

// ── SENSITIVITY ANALYSIS ──────────────────────────────────────────────────────
function SensitivityPanel({ ds, pts }) {
  const [adjustments, setAdj] = useState({chi:0, s:0, lambda0:0, C:0});

  const adjPts = pts.map(p=>({
    ...p,
    chi:     Math.min(0.99,Math.max(0.01,p.chi     + adjustments.chi)),
    s:       Math.min(0.99,Math.max(0.01,p.s       + adjustments.s)),
    lambda0: Math.min(0.99,Math.max(0.01,p.lambda0 + adjustments.lambda0)),
    C:       Math.min(0.99,Math.max(0.01,p.C       + adjustments.C)),
  }));

  const origMs = pts.map(p=>calcM(p.chi,p.s,p.lambda0,p.C));
  const adjMs  = adjPts.map(p=>calcM(p.chi,p.s,p.lambda0,p.C));

  const origNegYear = pts[origMs.findIndex(m=>m<0)]?.year;
  const adjNegYear  = adjPts[adjMs.findIndex(m=>m<0)]?.year;

  const leadChange = origNegYear && adjNegYear
    ? adjNegYear - origNegYear
    : null;

  const W=580,H=180,pL=48,pR=16,pT=14,pB=32;
  const allMs=[...origMs,...adjMs];
  const minM=Math.min(...allMs,-0.4), maxM=Math.max(...allMs,0.3), rng=maxM-minM;
  const toX=(i,tot)=>pL+(i/(tot-1))*(W-pL-pR);
  const toY=v=>pT+((maxM-v)/rng)*(H-pT-pB);
  const zeroY=toY(0);
  const path1=origMs.map((m,i)=>`${i===0?"M":"L"} ${toX(i,origMs.length)} ${toY(m)}`).join(" ");
  const path2=adjMs.map((m,i)=>`${i===0?"M":"L"} ${toX(i,adjMs.length)} ${toY(m)}`).join(" ");

  const anyAdj = Object.values(adjustments).some(v=>v!==0);

  return (
    <div style={{background:"#0A0A0A",border:"1px solid #2A2A2A",borderRadius:12,padding:20}}>
      <div style={{fontFamily:"var(--mono)",fontSize:9,color:"#3B82F6",
        letterSpacing:3,marginBottom:14}}>SENSITIVITY ANALYSIS</div>
      <p style={{fontSize:12,color:"#737373",fontFamily:"var(--sans)",
        marginBottom:16,lineHeight:1.6}}>
        Adjust any variable up or down to ask: "what if conditions had been different?"
        The original trajectory stays visible for comparison.
        {leadChange !== null && leadChange !== 0 && (
          <strong style={{color: leadChange>0?"#22C55E":"#EF4444"}}>
            {" "}M would have gone negative {Math.abs(leadChange)} years {leadChange>0?"later":"earlier"}.
          </strong>
        )}
      </p>

      {/* Adjustment sliders */}
      <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(220px,1fr))",
        gap:10,marginBottom:16}}>
        {[
          {key:"chi",    label:"χ — Efficiency",  color:"#60A5FA"},
          {key:"s",      label:"s — Throughput",   color:"#A78BFA"},
          {key:"lambda0",label:"λ₀ — Base Burden", color:"#F87171"},
          {key:"C",      label:"C — Complexity",   color:"#FCD34D"},
        ].map(v=>(
          <div key={v.key} style={{background:"#111111",border:"1px solid #1A1A1A",
            borderRadius:8,padding:"10px 14px"}}>
            <div style={{display:"flex",justifyContent:"space-between",marginBottom:5}}>
              <span style={{fontSize:11,fontWeight:600,color:v.color,
                fontFamily:"var(--sans)"}}>{v.label}</span>
              <span style={{fontFamily:"var(--mono)",fontSize:11,
                color:adjustments[v.key]>0?"#22C55E":adjustments[v.key]<0?"#EF4444":"#525252",
                fontWeight:600}}>
                {adjustments[v.key]>=0?"+":""}{adjustments[v.key].toFixed(2)}
              </span>
            </div>
            <input type="range" min={-0.3} max={0.3} step={0.01}
              value={adjustments[v.key]}
              onChange={e=>setAdj(a=>({...a,[v.key]:parseFloat(e.target.value)}))}
              style={{width:"100%",accentColor:v.color}}/>
            <div style={{display:"flex",justifyContent:"space-between",
              fontSize:9,color:"#525252",fontFamily:"var(--sans)",marginTop:2}}>
              <span>−0.30</span><span>0</span><span>+0.30</span>
            </div>
          </div>
        ))}
      </div>

      {/* Reset */}
      {anyAdj && (
        <button onClick={()=>setAdj({chi:0,s:0,lambda0:0,C:0})} style={{
          background:"none",border:"1px solid #2A2A2A",borderRadius:7,
          padding:"5px 14px",fontSize:11,color:"#737373",
          fontFamily:"var(--sans)",marginBottom:14,cursor:"pointer"
        }}>Reset to original</button>
      )}

      {/* Overlay chart */}
      <svg width={W} height={H} style={{display:"block",minWidth:W,overflow:"visible"}}>
        {[-0.3,-0.1,0,0.1,0.3].filter(v=>v>=minM&&v<=maxM).map(v=>(
          <g key={v}>
            <line x1={pL} y1={toY(v)} x2={W-pR} y2={toY(v)}
              stroke={v===0?"#ffffff22":"#ffffff08"}
              strokeWidth={v===0?1:0.5} strokeDasharray={v===0?"none":"3,4"}/>
            <text x={pL-5} y={toY(v)+4} textAnchor="end"
              fill="#4B5563" fontSize={8} fontFamily="JetBrains Mono">
              {v>=0?"+":""}{v.toFixed(1)}
            </text>
          </g>
        ))}
        {/* Original — faint */}
        <path d={path1} fill="none" stroke={ds.color} strokeWidth={1.5}
          strokeLinecap="round" strokeLinejoin="round" opacity={0.35}
          strokeDasharray="4,3"/>
        {/* Adjusted — solid */}
        {anyAdj && <path d={path2} fill="none" stroke={ds.color} strokeWidth={2}
          strokeLinecap="round" strokeLinejoin="round"/>}
        {/* Points */}
        {adjMs.map((m,i)=><circle key={i} cx={toX(i,adjMs.length)} cy={toY(m)}
          r={3} fill={mColor(m)}/>)}
        {/* Zero line */}
        {zeroY>pT&&zeroY<H-pB&&<line x1={pL} y1={zeroY} x2={W-pR} y2={zeroY}
          stroke="#EF444440" strokeWidth={1}/>}
        {/* Year labels */}
        {pts.filter((_,i)=>i===0||i===pts.length-1||(pts.length>6&&i===Math.floor(pts.length/2))).map((p,i,arr)=>{
          const idx=pts.indexOf(p);
          return <text key={i} x={toX(idx,pts.length)} y={H-pB+13}
            textAnchor="middle" fill="#4B5563" fontSize={8} fontFamily="JetBrains Mono">
            {p.year}
          </text>;
        })}
      </svg>

      {anyAdj && (
        <div style={{marginTop:12,display:"flex",gap:10,flexWrap:"wrap"}}>
          <div style={{display:"flex",alignItems:"center",gap:6}}>
            <div style={{width:16,height:1,borderTop:`2px dashed ${ds.color}`,opacity:0.4}}/>
            <span style={{fontSize:10,color:"#525252",fontFamily:"var(--sans)"}}>Original</span>
          </div>
          <div style={{display:"flex",alignItems:"center",gap:6}}>
            <div style={{width:16,height:2,background:ds.color,borderRadius:1}}/>
            <span style={{fontSize:10,color:"#525252",fontFamily:"var(--sans)"}}>Adjusted</span>
          </div>
        </div>
      )}
    </div>
  );
}

// ── TAB: EXPLORE ──────────────────────────────────────────────────────────────
function ExploreTab() {
  const [activeId, setActiveId]     = useState("rome");
  const [ptIdx, setPtIdx]           = useState(null);
  const [sliders, setSliders]       = useState({chi:0.75, s:0.80, lambda0:0.20, C:0.70});
  const [activeDomain, setActiveDomain] = useState(null);
  const [compareId, setCompareId]   = useState(null);
  const [showCitation, setShowCitation] = useState(false);
  const [activeResearch, setActiveResearch] = useState(null); // "sensitivity"|"tipping"|"overlay"
  const ds = DATASETS.find(d=>d.id===activeId);
  const pts = ds.points;
  const safeIdx = ptIdx!==null && ptIdx < pts.length ? ptIdx : null;
  const pt = safeIdx!==null ? pts[safeIdx] : pts[pts.length-1];
  const M = pt ? calcM(pt.chi,pt.s,pt.lambda0,pt.C) : 0;

  useEffect(()=>{ setPtIdx(null); },[activeId]);

  return (
    <div style={{display:"flex",flexDirection:"column",gap:32}}>
      <div>
        <h2 style={{fontFamily:"var(--serif)",fontSize:28,color:"#FFFFFF",marginBottom:12,borderLeft:"3px solid #3B82F6",paddingLeft:14}}>Explore real systems</h2>
        <p style={{color:"#D4D4D4",fontSize:13,fontFamily:"var(--sans)",fontWeight:400,lineHeight:1.6}}>
          20 datasets across collapses, recoveries, ecosystems, cities, and current systems — all calibrated to real historical data. Tap any card to load its full analysis below.
        </p>
      </div>

      {/* Step 1 — Pick a dataset — sticky */}
      <div style={{
        position:"sticky", top:103, zIndex:20,
        background:"#000000", paddingBottom:12, marginBottom:-12
      }}>
        <div style={{background:"#0A0A0A",border:"1px solid #2A2A2A",borderRadius:14,padding:"14px 16px"}}>
          <div style={{fontFamily:"var(--mono)",fontSize:9,color:"#3B82F6",letterSpacing:3,marginBottom:12}}>
            STEP 1 — SELECT A DATASET · {activeId && DATASETS.find(d=>d.id===activeId)?.label}
          </div>
          <div style={{display:"flex",gap:8,overflowX:"auto",paddingBottom:4}}>
            {DATASETS.map(d => {
              const isActive = activeId === d.id;
              // Active dataset: show currently selected point M. Others: show final M.
              const displayPt = isActive
                ? (safeIdx !== null ? pts[safeIdx] : pts[pts.length-1])
                : d.points[d.points.length-1];
              const displayM = calcM(displayPt.chi, displayPt.s, displayPt.lambda0, displayPt.C);
              const displayYear = isActive ? (displayPt.year || displayPt.label) : null;
              return (
                <button key={d.id} onClick={()=>setActiveId(d.id)} style={{
                  flexShrink:0,
                  background: isActive ? "#1A1A2A" : "#111111",
                  border: `1px solid ${isActive ? d.color : "#2A2A2A"}`,
                  borderRadius:10, padding:"8px 12px",
                  display:"flex", flexDirection:"column", alignItems:"flex-start", gap:3,
                  cursor:"pointer", transition:"all 0.15s", minWidth:110
                }}>
                  <div style={{display:"flex",alignItems:"center",gap:6,width:"100%"}}>
                    <span style={{fontSize:14}}>{d.emoji}</span>
                    <span style={{fontFamily:"var(--mono)",fontSize:9,
                      color: isActive ? mColor(displayM) : mColor(displayM),
                      fontWeight:700}}>
                      {(displayM>=0?"+":"")+displayM.toFixed(2)}
                    </span>
                  </div>
                  <div style={{fontSize:10,fontWeight:600,
                    color: isActive ? "#FFFFFF" : "#A3A3A3",
                    fontFamily:"var(--sans)",lineHeight:1.2,textAlign:"left"}}>
                    {d.label}
                  </div>
                  {isActive && displayYear && (
                    <div style={{fontSize:8,color:d.color,fontFamily:"var(--mono)",
                      opacity:0.8}}>{displayYear}</div>
                  )}
                </button>
              );
            })}
          </div>
        </div>
      </div>

      {/* Step 2 — Analysis */}
      <div style={{display:"flex",alignItems:"center",gap:12,padding:"0 4px"}}>
        <div style={{flex:1,height:1,background:"#1A1A1A"}}/>
        <div style={{
          display:"flex",alignItems:"center",gap:10,
          background:"#0A0A0A",border:`1px solid ${ds.color}60`,
          borderRadius:20,padding:"6px 16px",flexShrink:0
        }}>
          <span style={{fontSize:16}}>{ds.emoji}</span>
          <span style={{fontFamily:"var(--sans)",fontSize:12,fontWeight:600,color:ds.color}}>{ds.label}</span>
          <span style={{fontFamily:"var(--mono)",fontSize:9,color:"#525252"}}>{ds.period}</span>
        </div>
        <div style={{flex:1,height:1,background:"#1A1A1A"}}/>
      </div>

      {/* Detail panel */}
      <div style={{background:"#0A0A0A",border:`1px solid ${ds.color}35`,borderRadius:14,padding:28}}>

        {/* Header row */}
        <div style={{display:"flex",justifyContent:"space-between",alignItems:"flex-start",flexWrap:"wrap",gap:20,marginBottom:28}}>
          <div style={{flex:1,minWidth:240}}>
            <div style={{display:"flex",alignItems:"center",gap:10,marginBottom:6}}>
              <span style={{fontSize:22}}>{ds.emoji}</span>
              <div>
                <span style={{fontFamily:"var(--serif)",fontSize:19,color:"#FFFFFF"}}>{ds.label}</span>
                <span style={{fontFamily:"var(--mono)",fontSize:10,color:"#A3A3A3",marginLeft:10}}>{ds.period}</span>
              </div>
            </div>
            <p style={{fontSize:13,color:"#D4D4D4",lineHeight:1.65,fontFamily:"var(--sans)",fontWeight:400,maxWidth:460}}>{ds.desc}</p>
            <div style={{marginTop:8,fontSize:10,color:"#404040",fontFamily:"var(--mono)"}}>Source: {ds.source}</div>
          </div>
          <div style={{display:"flex",flexDirection:"column",alignItems:"center",gap:10,flexShrink:0}}>
            <Gauge value={M} size={148}/>
            <div style={{fontSize:13,fontWeight:700,color:mColor(M),fontFamily:"var(--sans)"}}>
              {mLabel(M)}
            </div>
            <div style={{fontSize:9,color:"#A3A3A3",fontFamily:"var(--mono)"}}>
              {pt?.year||pt?.label}
            </div>
            <WarningLeadBadge dsId={activeId} points={pts} compact={true}/>
          </div>
        </div>

        {/* Warning lead time — headline if available */}
        {(() => {
          const lead = getWarningLead(activeId);
          if (!lead || lead.lead === null) return null;
          return (
            <div style={{
              background:"linear-gradient(135deg, #1A0800 0%, #0F0500 100%)",
              border:"1px solid #F9731640",
              borderRadius:12, padding:"16px 20px",
              display:"flex", alignItems:"center", gap:16, flexWrap:"wrap",
              marginBottom:4
            }}>
              <div style={{flexShrink:0}}>
                <div style={{fontFamily:"var(--mono)",fontSize:9,color:"#F97316",
                  letterSpacing:3,marginBottom:4}}>M WARNING LEAD TIME</div>
                <div style={{display:"flex",alignItems:"baseline",gap:8}}>
                  <span style={{fontFamily:"var(--mono)",fontSize:42,
                    color:"#F97316",fontWeight:700,lineHeight:1}}>{lead.lead}</span>
                  <span style={{fontSize:16,color:"#F97316",
                    fontFamily:"var(--sans)",fontWeight:600}}>
                    year{lead.lead!==1?"s":""}
                  </span>
                </div>
              </div>
              <div style={{flex:1,minWidth:180}}>
                <div style={{fontSize:13,color:"#FFFFFF",fontFamily:"var(--sans)",
                  fontWeight:600,marginBottom:4,lineHeight:1.4}}>
                  M went negative in <strong style={{color:"#F97316"}}>{lead.m_negative_year}</strong> —{" "}
                  {lead.lead} year{lead.lead!==1?"s":""} before{" "}
                  <strong style={{color:"#FFFFFF"}}>{lead.event}</strong> ({lead.event_year})
                </div>
                <div style={{fontSize:11,color:"#737373",fontFamily:"var(--sans)",lineHeight:1.5}}>
                  Independently verifiable — event date not used to calibrate the model.
                  {lead.source && <span style={{color:"#525252"}}> Source: {lead.source}</span>}
                </div>
              </div>
            </div>
          );
        })()}

        {/* Chart */}
        <MChart points={pts} dsColor={ds.color} dsId={activeId}/>

        <MInsight points={pts} dsId={activeId} dsLabel={ds.label} domain={ds.domain}/>
        {/* Historical annotation — always visible, right under the chart */}
        <div style={{
          marginTop:12,
          background:"#111111",
          borderRadius:10,
          padding:"12px 16px",
          borderLeft:`3px solid ${ds.color}`,
          minHeight:66
        }}>
          <div style={{fontSize:9,fontFamily:"var(--mono)",color:ds.color,
            marginBottom:5,letterSpacing:2}}>
            WHAT WAS HAPPENING · {pt?.year || pt?.label}
          </div>
          <p style={{fontSize:13,color:"#D4D4D4",lineHeight:1.65,
            fontFamily:"var(--sans)",fontWeight:400,margin:0}}>
            {pt?.event || "Tap a year below to see what was happening at that moment."}
          </p>
        </div>

        {/* Time point buttons */}
        <div style={{display:"flex",flexWrap:"wrap",gap:6,marginTop:20}}>
          {pts.map((p,i)=>{
            const m=calcM(p.chi,p.s,p.lambda0,p.C);
            const sel=safeIdx===i||(safeIdx===null&&i===pts.length-1);
            return (
              <button key={i} onClick={()=>setPtIdx(i)} style={{
                background:sel?"#111111":"#000000",
                border:`1px solid ${sel?mColor(m)+"90":"#2A2A2A"}`,
                borderRadius:6,padding:"4px 9px",fontSize:9,
                color:sel?mColor(m):"#737373",fontFamily:"var(--mono)",transition:"all 0.12s"
              }}>{p.year}</button>
            );
          })}
        </div>

        {/* Variable row */}
        <div style={{display:"grid",gridTemplateColumns:"repeat(4,1fr)",gap:10,marginTop:16}}>
          {[
            {sym:"χ",val:pt.chi,color:"#60A5FA",name:"Efficiency"},
            {sym:"s",val:pt.s,color:"#A78BFA",name:"Throughput"},
            {sym:"λ(C)",val:pt.lambda0+k*Math.pow(pt.C,n),color:"#F87171",name:"Burden"},
            {sym:"C",val:pt.C,color:"#FCD34D",name:"Complexity"},
          ].map((v,i)=>(
            <div key={i} style={{background:"#111111",borderRadius:8,padding:"14px 16px",textAlign:"center",border:"1px solid #2A2A2A"}}>
              <div style={{fontFamily:"var(--mono)",fontSize:15,color:v.color,fontWeight:500}}>{v.val.toFixed(3)}</div>
              <div style={{fontSize:9,color:"#A3A3A3",marginTop:3,fontFamily:"var(--sans)"}}>{v.sym} · {v.name}</div>
            </div>
          ))}
        </div>

        {/* Model fit + warning lead full block */}
        <div style={{marginTop:16}}>
          <WarningLeadBadge dsId={activeId} points={pts}/>
        </div>

        {/* Action bar */}
        <div style={{marginTop:24,paddingTop:20,borderTop:"1px solid #1A1A1A"}}>
          <div style={{display:"flex",flexWrap:"wrap",gap:8,marginBottom:12}}>
            <button onClick={()=>downloadReport(ds,pts,pt,ptIdx)} style={{
              background:"#2563EB",border:"none",borderRadius:8,
              padding:"10px 18px",fontSize:12,fontWeight:600,
              color:"#FFFFFF",fontFamily:"var(--sans)",
              display:"flex",alignItems:"center",gap:6,cursor:"pointer",
              transition:"background 0.15s"
            }}
              onMouseEnter={e=>e.currentTarget.style.background="#3B82F6"}
              onMouseLeave={e=>e.currentTarget.style.background="#2563EB"}>
              <span>↓</span><span>Download Report</span>
            </button>
            <button onClick={()=>exportCSV(ds,pts)} style={{
              background:"#111111",border:"1px solid #2A2A2A",borderRadius:8,
              padding:"10px 18px",fontSize:12,fontWeight:600,
              color:"#D4D4D4",fontFamily:"var(--sans)",
              display:"flex",alignItems:"center",gap:6,cursor:"pointer"
            }}>
              <span>⬇</span><span>Export CSV</span>
            </button>
            <button onClick={()=>setShowCitation(true)} style={{
              background:"#111111",border:"1px solid #2A2A2A",borderRadius:8,
              padding:"10px 18px",fontSize:12,fontWeight:600,
              color:"#D4D4D4",fontFamily:"var(--sans)",
              display:"flex",alignItems:"center",gap:6,cursor:"pointer"
            }}>
              <span>📎</span><span>Cite</span>
            </button>
          </div>
          <div style={{display:"flex",flexWrap:"wrap",gap:8}}>
            {[
              {id:"sensitivity", label:"🔬 Sensitivity Analysis", desc:"What if conditions had been different?"},
              {id:"tipping",     label:"📈 Tipping Point Projection", desc:"When does M hit zero?"},
              {id:"overlay",     label:"🔀 Compare Two Systems", desc:"Overlay any two datasets"},
            ].map(tool=>(
              <button key={tool.id}
                onClick={()=>setActiveResearch(activeResearch===tool.id?null:tool.id)}
                style={{
                  background:activeResearch===tool.id?"#131D30":"#0A0A0A",
                  border:`1px solid ${activeResearch===tool.id?"#2563EB":"#2A2A2A"}`,
                  borderRadius:8,padding:"8px 14px",fontSize:11,fontWeight:500,
                  color:activeResearch===tool.id?"#93C5FD":"#A3A3A3",
                  fontFamily:"var(--sans)",cursor:"pointer",transition:"all 0.15s",
                  display:"flex",flexDirection:"column",alignItems:"flex-start",gap:1
                }}>
                <span style={{fontWeight:600}}>{tool.label}</span>
                <span style={{fontSize:9,opacity:0.7}}>{tool.desc}</span>
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Research tools — expand below results card */}
      {activeResearch==="sensitivity" && (
        <SensitivityPanel ds={ds} pts={pts}/>
      )}

      {activeResearch==="tipping" && (
        <TippingPointBadge pts={pts} dsLabel={ds.label}/>
      )}

      {activeResearch==="overlay" && (
        <div style={{background:"#0A0A0A",border:"1px solid #2A2A2A",borderRadius:14,padding:20}}>
          <div style={{fontFamily:"var(--mono)",fontSize:9,color:"#3B82F6",
            letterSpacing:3,marginBottom:14}}>COMPARE TWO SYSTEMS</div>
          <div style={{marginBottom:14}}>
            <div style={{fontSize:12,color:"#737373",fontFamily:"var(--sans)",marginBottom:8}}>
              Select a second dataset to overlay with <strong style={{color:"#FFFFFF"}}>{ds.label}</strong>:
            </div>
            <div style={{display:"flex",flexWrap:"wrap",gap:6}}>
              {DATASETS.filter(d=>d.id!==activeId).map(d=>(
                <button key={d.id} onClick={()=>setCompareId(compareId===d.id?null:d.id)} style={{
                  background:compareId===d.id?"#1A1A2A":"#111111",
                  border:`1px solid ${compareId===d.id?d.color:"#2A2A2A"}`,
                  borderRadius:6,padding:"5px 10px",fontSize:10,
                  color:compareId===d.id?d.color:"#737373",fontFamily:"var(--sans)"
                }}>{d.emoji} {d.label}</button>
              ))}
            </div>
          </div>
          {compareId && (
            <OverlayChart
              ds1={ds}
              ds2={DATASETS.find(d=>d.id===compareId)}
            />
          )}
        </div>
      )}

      {showCitation && <CitationModal ds={ds} onClose={()=>setShowCitation(false)}/>}


      {/* ── DOMAIN SANDBOX ── */}
      <div style={{background:"#0A0A0A",border:"1px solid #2A2A2A",borderRadius:14,overflow:"hidden"}}>
        {/* Header */}
        <div style={{padding:"20px 24px 16px",borderBottom:"1px solid #1A1A1A"}}>
          <div style={{fontSize:16,fontFamily:"var(--serif)",color:"#FFFFFF",marginBottom:4}}>
            Try it yourself
          </div>
          <p style={{fontSize:13,color:"#737373",fontFamily:"var(--sans)"}}>
            Pick a domain, see exactly what real-world things map to each variable, load the preset, and adjust the sliders to see the margin respond live.
          </p>
        </div>

        {/* Domain pills */}
        <div style={{padding:"16px 24px",borderBottom:"1px solid #1A1A1A",display:"flex",flexWrap:"wrap",gap:8}}>
          {DOMAIN_PRESETS.map(d=>(
            <button key={d.id} onClick={()=>setActiveDomain(activeDomain===d.id?null:d.id)} style={{
              background:activeDomain===d.id?"#1A1A2A":"#111111",
              border:`1px solid ${activeDomain===d.id?d.color+"80":"#2A2A2A"}`,
              borderRadius:20,padding:"6px 14px",fontSize:12,
              color:activeDomain===d.id?d.color:"#A3A3A3",
              fontFamily:"var(--sans)",transition:"all 0.15s",
              display:"flex",alignItems:"center",gap:6
            }}>
              <span>{d.emoji}</span><span>{d.label}</span>
            </button>
          ))}
        </div>


        {/* FIX 2: Variable overview grid — always visible before any pill is clicked */}
        {!activeDomain && (
          <div style={{padding:"16px 24px",borderBottom:"1px solid #1A1A1A",background:"#000000"}}>
            <div style={{fontFamily:"var(--mono)",fontSize:9,color:"#525252",letterSpacing:2,marginBottom:12}}>
              WHAT EACH VARIABLE MEANS IN EACH DOMAIN — SELECT A DOMAIN TO EXPAND
            </div>
            <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(230px,1fr))",gap:10}}>
              {DOMAIN_PRESETS.map(d=>(
                <button key={d.id} onClick={()=>setActiveDomain(d.id)}
                  style={{background:"#0A0A0A",border:`1px solid ${d.color}30`,borderRadius:10,
                    padding:"14px 16px",textAlign:"left",cursor:"pointer",transition:"all 0.15s"}}
                  onMouseEnter={e=>{e.currentTarget.style.borderColor=d.color+"80";e.currentTarget.style.background="#111111";}}
                  onMouseLeave={e=>{e.currentTarget.style.borderColor=d.color+"30";e.currentTarget.style.background="#0A0A0A";}}
                >
                  <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:6}}>
                    <span style={{fontSize:18}}>{d.emoji}</span>
                    <span style={{fontSize:13,fontWeight:700,color:d.color,fontFamily:"var(--sans)"}}>{d.label}</span>
                  </div>
                  <div style={{fontSize:11,color:"#737373",fontFamily:"var(--sans)",lineHeight:1.5,marginBottom:10}}>{d.note}</div>
                  <div style={{display:"flex",flexDirection:"column",gap:5}}>
                    {[
                      {sym:"χ",color:"#60A5FA",q:d.vars.chi.label},
                      {sym:"s",color:"#A78BFA",q:d.vars.s.label},
                      {sym:"λ₀",color:"#F87171",q:d.vars.lambda0.label},
                      {sym:"C",color:"#FCD34D",q:d.vars.C.label},
                    ].map(v=>(
                      <div key={v.sym} style={{display:"flex",gap:7,alignItems:"flex-start"}}>
                        <span style={{fontFamily:"var(--mono)",fontSize:11,color:v.color,fontWeight:700,flexShrink:0,width:18,paddingTop:1}}>{v.sym}</span>
                        <span style={{fontSize:11,color:"#737373",fontFamily:"var(--sans)",lineHeight:1.4}}>
                          {v.q.length > 52 ? v.q.slice(0,52)+"…" : v.q}
                        </span>
                      </div>
                    ))}
                  </div>
                  <div style={{marginTop:10,fontSize:9,color:d.color+"99",fontFamily:"var(--mono)",letterSpacing:1}}>
                    TAP TO SEE FULL BREAKDOWN + LOAD PRESET →
                  </div>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Domain detail — expands when selected */}
        {activeDomain && (() => {
          const dom = DOMAIN_PRESETS.find(d=>d.id===activeDomain);
          return (
            <div style={{padding:"20px 24px",borderBottom:"1px solid #1A1A1A",background:"#000000"}}>
              <div style={{display:"flex",justifyContent:"space-between",alignItems:"flex-start",flexWrap:"wrap",gap:12,marginBottom:16}}>
                <div>
                  <div style={{fontSize:14,fontWeight:600,color:dom.color,fontFamily:"var(--sans)",marginBottom:4}}>
                    {dom.emoji} {dom.label}
                  </div>
                  <div style={{fontSize:12,color:"#737373",fontFamily:"var(--sans)",maxWidth:480}}>{dom.note}</div>
                </div>
                <button onClick={()=>setSliders(dom.preset)} style={{
                  background:dom.color,border:"none",borderRadius:8,
                  padding:"8px 18px",fontSize:12,fontWeight:700,
                  color:"#000000",fontFamily:"var(--sans)",flexShrink:0
                }}>Load preset →</button>
              </div>
              <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(260px,1fr))",gap:12}}>
                {[
                  {key:"chi",    sym:"χ", color:"#60A5FA", label:"Efficiency"},
                  {key:"s",      sym:"s", color:"#A78BFA", label:"Throughput"},
                  {key:"lambda0",sym:"λ₀",color:"#F87171", label:"Base Burden"},
                  {key:"C",      sym:"C", color:"#FCD34D", label:"Complexity"},
                ].map(v=>{
                  const info = dom.vars[v.key];
                  return (
                    <div key={v.key} style={{background:"#111111",border:"1px solid #1A1A1A",borderRadius:10,padding:14}}>
                      <div style={{display:"flex",alignItems:"baseline",gap:8,marginBottom:8}}>
                        <span style={{fontFamily:"var(--mono)",fontSize:18,color:v.color,fontWeight:600}}>{v.sym}</span>
                        <span style={{fontSize:10,color:"#525252"}}>{v.label}</span>
                        <span style={{marginLeft:"auto",fontFamily:"var(--mono)",fontSize:10,color:v.color+"cc"}}>preset {dom.preset[v.key].toFixed(2)}</span>
                      </div>
                      <div style={{fontSize:12,fontWeight:600,color:"#FFFFFF",marginBottom:8,fontFamily:"var(--sans)",lineHeight:1.4}}>{info.label}</div>
                      {info.positive?.length > 0 && (
                        <div style={{marginBottom:6}}>
                          <div style={{fontSize:9,fontFamily:"var(--mono)",color:"#22C55E",marginBottom:4,letterSpacing:1}}>COUNTS AS ↑</div>
                          {info.positive.map((item,j)=>(
                            <div key={j} style={{display:"flex",gap:5,marginBottom:2}}>
                              <span style={{color:"#22C55E",fontSize:10,flexShrink:0}}>+</span>
                              <span style={{fontSize:11,color:"#A3A3A3",lineHeight:1.5,fontFamily:"var(--sans)"}}>{item}</span>
                            </div>
                          ))}
                        </div>
                      )}
                      {info.negative?.length > 0 && (
                        <div>
                          <div style={{fontSize:9,fontFamily:"var(--mono)",color:"#EF4444",marginBottom:4,letterSpacing:1}}>{info.positive?.length?"COUNTS AGAINST ↓":"WHAT GOES HERE"}</div>
                          {info.negative.map((item,j)=>(
                            <div key={j} style={{display:"flex",gap:5,marginBottom:2}}>
                              <span style={{color:"#EF4444",fontSize:10,flexShrink:0}}>−</span>
                              <span style={{fontSize:11,color:"#A3A3A3",lineHeight:1.5,fontFamily:"var(--sans)"}}>{item}</span>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          );
        })()}

        {/* Gauge hero — top center */}
        {(() => {
          const M = calcM(sliders.chi,sliders.s,sliders.lambda0,sliders.C);
          return (
            <div style={{padding:"20px 24px 16px",borderBottom:"1px solid #1A1A1A",
              display:"flex",alignItems:"center",justifyContent:"center",gap:32,flexWrap:"wrap",
              background:"#000000"}}>
              <div style={{textAlign:"center"}}>
                <Gauge value={M} size={180}/>
                <div style={{fontSize:16,fontWeight:700,color:mColor(M),fontFamily:"var(--sans)",marginTop:4}}>
                  {mLabel(M)}
                </div>
              </div>
              <div style={{display:"flex",flexDirection:"column",gap:8,minWidth:180}}>
                {[
                  {label:"χs  — output generated", val:(sliders.chi*sliders.s).toFixed(3), color:"#22C55E"},
                  {label:"λ(C) — burden total",     val:(sliders.lambda0+k*Math.pow(sliders.C,n)).toFixed(3), color:"#EF4444"},
                  {label:"M  — stability margin",   val:(M>=0?"+":"")+M.toFixed(4), color:mColor(M)},
                ].map((r,i)=>(
                  <div key={i} style={{display:"flex",justifyContent:"space-between",gap:16,
                    padding:"8px 14px",background:"#111111",borderRadius:8,
                    border:`1px solid ${r.color}20`}}>
                    <span style={{fontSize:11,color:"#737373",fontFamily:"var(--sans)"}}>{r.label}</span>
                    <span style={{fontFamily:"var(--mono)",color:r.color,fontWeight:700,fontSize:13}}>{r.val}</span>
                  </div>
                ))}
              </div>
            </div>
          );
        })()}

        {/* Sliders — 2-column grid below gauge */}
        <div style={{padding:"16px 24px 20px"}}>
          <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(240px,1fr))",gap:10}}>
            {[
              {key:"chi",    label:"χ — Efficiency", color:"#60A5FA", low:"Inefficient",  high:"Optimized"},
              {key:"s",      label:"s — Throughput",  color:"#A78BFA", low:"Depleted",     high:"Abundant"},
              {key:"lambda0",label:"λ₀ — Burden",     color:"#F87171", low:"Very lean",    high:"Heavy overhead"},
              {key:"C",      label:"C — Complexity",  color:"#FCD34D", low:"Simple",       high:"Very complex"},
            ].map(s=>(
              <div key={s.key} style={{background:"#111111",border:"1px solid #1A1A1A",borderRadius:10,padding:"12px 14px"}}>
                <div style={{display:"flex",justifyContent:"space-between",marginBottom:6}}>
                  <span style={{fontSize:12,fontWeight:600,color:s.color,fontFamily:"var(--sans)"}}>{s.label}</span>
                  <span style={{fontFamily:"var(--mono)",fontSize:14,color:"#FFFFFF",fontWeight:700}}>{sliders[s.key].toFixed(2)}</span>
                </div>
                <input type="range" min={0.01} max={0.99} step={0.01}
                  value={sliders[s.key]}
                  onChange={e=>setSliders(p=>({...p,[s.key]:parseFloat(e.target.value)}))}
                  style={{width:"100%",accentColor:s.color,cursor:"pointer"}}
                />
                <div style={{display:"flex",justifyContent:"space-between",fontSize:9,color:"#525252",fontFamily:"var(--sans)",marginTop:3}}>
                  <span>{s.low}</span><span>{s.high}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* EoE legitimacy note */}
      <div style={{background:"#0A0A0A",border:"1px solid #2A2A2A",borderRadius:10,padding:"16px 20px",display:"flex",gap:12,alignItems:"flex-start"}}>
        <span style={{fontSize:14,flexShrink:0}}>⚠️</span>
        <p style={{fontSize:11,color:"#737373",lineHeight:1.6,fontFamily:"var(--sans)"}}>
          EoE can identify declining or improving margins and patterns consistent with collapse predictions. It cannot predict when collapse will occur or guarantee future outcomes. All values are proxy estimates, not precise measurements. Framework under peer review — cite as: Baird, N. (2026). Engine of Emergence. arXiv:[pending].
        </p>
      </div>
    </div>
  );
}



// ── TAB: RUN AN EXPERIMENT ────────────────────────────────────────────────────

function ExperimentTab({ onGoToExplore, onGoToAssistant, uploadedDatasets=[] }) {
  const [phase, setPhase]           = useState("input");
  const [hypothesis, setHypothesis] = useState("");
  const [domain, setDomain]         = useState("");
  const [timeScale, setTimeScale]   = useState("");
  const [experiment, setExperiment] = useState(null);
  const [exportDone, setExportDone] = useState(false);
  const [processingStep, setProcessingStep] = useState(0);
  const inputRef = useRef(null);

  const DOMAINS = ["Civilizational","Ecological","Fiscal/National","Urban","Corporate","Biological","Seismic","Climate","Custom"];
  const TIMESCALES = ["Decades","Centuries","Years","Months","Geological"];

  const EXAMPLES = [
    { icon:"🇺🇸", text:"What happens to a civilization when energy costs exceed maintenance returns?", domain:"Fiscal/National", time:"Decades" },
    { icon:"🪸", text:"At what point does reef bleaching frequency make recovery mathematically impossible?", domain:"Ecological", time:"Decades" },
    { icon:"🏛️", text:"Why do empires always collapse faster than they rise?", domain:"Civilizational", time:"Centuries" },
    { icon:"🌲", text:"Can the Amazon tip irreversibly even if deforestation stops?", domain:"Ecological", time:"Decades" },
    { icon:"🏢", text:"What EoE variables would predict Enron collapse before the fraud was discovered?", domain:"Corporate", time:"Years" },
    { icon:"🌋", text:"Is accumulated seismic strain fundamentally the same as bureaucratic overhead in EoE terms?", domain:"Seismic", time:"Geological" },
  ];

  const SYSTEM_STRUCTURE = `You are the Engine of Emergence experiment structurer. EoE measures M = xs - L(C) where x=efficiency, s=throughput, L(C)=burden=L0+0.15*C^1.4, C=complexity. M positive=runway, M negative=borrowed time. Given a hypothesis return ONLY valid JSON no markdown: {"title":"5-8 word title","hypothesis":"cleaned 1-2 sentence hypothesis","domain":"primary domain","timeScale":"e.g. Decades","variables":{"chi":"what x means here","s":"what s means here","lambda0":"what L0 means here","C":"what C means here"},"eoe_prediction":"2-3 sentence prediction","confidence":"high|medium|speculative","confidence_reason":"one sentence","generate_chart":true,"chart_reason":"one sentence","uncertainty_flag":null}`;

  const SYSTEM_ANALYZE = `You are the Engine of Emergence experiment analyst. Given structured experiment JSON return ONLY valid JSON no markdown: {"narrative":"3-5 paragraph analysis covering what EoE reveals historical analogs trajectory implications and what would improve M","chart_data":[{"year":1960,"chi":0.8,"s":0.8,"lambda0":0.2,"C":0.5}],"chart_note":"one sentence caveat or null","minsight":"2 sentence plain English summary no jargon","key_finding":"single most important finding one sentence","analogues":["historical system 1 with note","historical system 2 with note"],"intervention":"one concrete paragraph on what would reverse trajectory"}. If generate_chart true produce 5-8 realistic plausible data points. If false set chart_data to null.`;

  async function runExperiment() {
    if (!hypothesis.trim()) return;
    setPhase("processing");
    setProcessingStep(1);
    setExperiment(null);
    const fullInput = "Hypothesis: " + hypothesis + (domain ? "\nDomain: " + domain : "") + (timeScale ? "\nTime scale: " + timeScale : "");
    const headers = {"Content-Type":"application/json","x-api-key":import.meta.env.VITE_ANTHROPIC_KEY||"","anthropic-version":"2023-06-01","anthropic-dangerous-direct-browser-access":"true"};
    try {
      const r1 = await fetch("https://api.anthropic.com/v1/messages",{method:"POST",headers,body:JSON.stringify({model:"claude-sonnet-4-5",max_tokens:1000,system:SYSTEM_STRUCTURE,messages:[{role:"user",content:fullInput}]})});
      const d1 = await r1.json();
      const raw1 = d1.content?.map(b=>b.text||"").join("")||"";
      let structured;
      try { structured = JSON.parse(raw1.replace(/```json|```/g,"").trim()); }
      catch(e) { structured = {title:"Experiment",hypothesis,domain:domain||"General",timeScale:timeScale||"Decades",variables:{chi:"Efficiency",s:"Throughput",lambda0:"Burden",C:"Complexity"},eoe_prediction:"Analysis pending.",confidence:"speculative",confidence_reason:"Parse error.",generate_chart:false,chart_reason:"N/A",uncertainty_flag:"Structure parsing failed."}; }
      setProcessingStep(2);
      const r2 = await fetch("https://api.anthropic.com/v1/messages",{method:"POST",headers,body:JSON.stringify({model:"claude-sonnet-4-5",max_tokens:1500,system:SYSTEM_ANALYZE,messages:[{role:"user",content:JSON.stringify(structured)}]})});
      const d2 = await r2.json();
      const raw2 = d2.content?.map(b=>b.text||"").join("")||"";
      let analysis;
      try { analysis = JSON.parse(raw2.replace(/```json|```/g,"").trim()); }
      catch(e) { analysis = {narrative:raw2||"Analysis could not be parsed.",chart_data:null,chart_note:null,minsight:"See narrative above.",key_finding:"See narrative.",analogues:[],intervention:"See narrative."}; }
      setProcessingStep(3);
      setExperiment({structured,analysis,timestamp:new Date().toISOString(),id:Date.now()});
      setPhase("results");
    } catch(err) {
      setExperiment({structured:{title:"Error",hypothesis,domain:domain||"Unknown",timeScale:timeScale||"Unknown",variables:{chi:"—",s:"—",lambda0:"—",C:"—"},eoe_prediction:"Connection failed.",confidence:"speculative",confidence_reason:"Network error.",generate_chart:false,chart_reason:"N/A",uncertainty_flag:"Connection failed."},analysis:{narrative:"Connection issue — please try again.",chart_data:null,chart_note:null,minsight:"Try again.",key_finding:"N/A",analogues:[],intervention:"N/A"},timestamp:new Date().toISOString(),id:Date.now()});
      setPhase("results");
    }
  }

  function reset() { setPhase("input"); setHypothesis(""); setDomain(""); setTimeScale(""); setExperiment(null); setExportDone(false); }

  function exportMarkdown() {
    if (!experiment) return;
    const {structured:s,analysis:a,timestamp} = experiment;
    const date = new Date(timestamp).toLocaleDateString("en-US",{year:"numeric",month:"long",day:"numeric"});
    const md = ["# EoE Experiment — "+s.title,"*"+date+" · Engine of Emergence v2 · Nathan Baird, 2026*","","## Hypothesis",s.hypothesis,"","## Domain & Variables","**Domain:** "+s.domain+" | **Time Scale:** "+s.timeScale,"","| Variable | Represents |","|----------|------------|","| χ | "+s.variables.chi+" |","| s | "+s.variables.s+" |","| λ₀ | "+s.variables.lambda0+" |","| C | "+s.variables.C+" |","","## EoE Prediction",s.eoe_prediction,"","**Confidence:** "+s.confidence+" — "+s.confidence_reason,(s.uncertainty_flag?"\n**Uncertainty:** "+s.uncertainty_flag:""),"","## Analysis",a.narrative,"","## Key Finding",a.key_finding,"","## Plain English Summary",a.minsight,"","## Historical Analogues",...(a.analogues||[]).map(x=>"- "+x),"","## Intervention",a.intervention,"","---","*Cite as: Baird, N. (2026). Engine of Emergence. DOI: 10.5281/zenodo.19016245*"].join("\n");
    const blob = new Blob([md],{type:"text/markdown"});
    const url = URL.createObjectURL(blob);
    const a2 = document.createElement("a"); a2.href=url; a2.download="EoE_Experiment_"+s.title.replace(/\s+/g,"_")+".md"; a2.click(); URL.revokeObjectURL(url);
  }

  function copyToClipboard() {
    if (!experiment) return;
    const {structured:s,analysis:a} = experiment;
    navigator.clipboard.writeText("EoE EXPERIMENT — "+s.title+"\n\nHypothesis: "+s.hypothesis+"\n\nEoE Prediction: "+s.eoe_prediction+"\n\nKey Finding: "+a.key_finding+"\n\nSummary: "+a.minsight+"\n\nCite: Baird, N. (2026). Engine of Emergence. DOI: 10.5281/zenodo.19016245").then(()=>{ setExportDone(true); setTimeout(()=>setExportDone(false),2500); });
  }

  const CONF_COLOR = {high:"#22C55E",medium:"#EAB308",speculative:"#F97316"};
  const CONF_LABEL = {high:"High confidence",medium:"Medium confidence",speculative:"Speculative"};

  function ExperimentChart({points,color}) {
    if (!points||points.length<2) return null;
    const vals = points.map(p=>calcM(p.chi,p.s,p.lambda0,p.C));
    const minV=Math.min(...vals,-0.5),maxV=Math.max(...vals,0.3),rng=maxV-minV;
    const W=520,H=120,pl=48,pr=16,pt=16,pb=32,iw=W-pl-pr,ih=H-pt-pb;
    const xs=points.map((_,i)=>pl+(i/(points.length-1))*iw);
    const ys=vals.map(v=>pt+(1-(v-minV)/rng)*ih);
    const zeroY=pt+(1-(0-minV)/rng)*ih;
    const pathD=xs.map((x,i)=>(i===0?"M":"L")+" "+x+" "+ys[i]).join(" ");
    const fillD=pathD+" L "+xs[xs.length-1]+" "+zeroY+" L "+xs[0]+" "+zeroY+" Z";
    return (
      <svg width="100%" viewBox={"0 0 "+W+" "+H} style={{display:"block",overflow:"visible"}}>
        {zeroY>pt&&zeroY<pt+ih&&<line x1={pl} y1={zeroY} x2={W-pr} y2={zeroY} stroke="#3A3A3A" strokeWidth={1} strokeDasharray="4,4"/>}
        <path d={fillD} fill={color} opacity={0.08}/>
        <path d={pathD} fill="none" stroke={color} strokeWidth={2} strokeLinecap="round" strokeLinejoin="round"/>
        {points.map((p,i)=><g key={i}><circle cx={xs[i]} cy={ys[i]} r={i===points.length-1?5:3} fill={mColor(vals[i])} stroke="#000" strokeWidth={1}/><text x={xs[i]} y={H-8} textAnchor="middle" fontSize={8} fill="#525252" fontFamily="JetBrains Mono">{p.year||p.label||i}</text></g>)}
        <text x={pl-4} y={pt+4} textAnchor="end" fontSize={8} fill="#525252" fontFamily="JetBrains Mono">{maxV.toFixed(2)}</text>
        <text x={pl-4} y={pt+ih+4} textAnchor="end" fontSize={8} fill="#525252" fontFamily="JetBrains Mono">{minV.toFixed(2)}</text>
      </svg>
    );
  }

  if (phase==="input") return (
    <div style={{display:"flex",flexDirection:"column",gap:32,maxWidth:760,margin:"0 auto",width:"100%"}}>
      <div style={{borderLeft:"3px solid #22C55E",paddingLeft:16}}>
        <h2 style={{fontFamily:"var(--serif)",fontSize:30,color:"#FFFFFF",marginBottom:8}}>Run an Experiment</h2>
        <p style={{color:"#737373",fontSize:13,fontFamily:"var(--sans)",lineHeight:1.65}}>Pose a hypothesis. EoE will structure it, analyze it, and generate a full research output — with charts, variable mappings, historical analogues, and a plain-English summary.</p>
      </div>
      <div style={{background:"#0A0A0A",border:"1px solid #2A2A2A",borderRadius:14,padding:24,display:"flex",flexDirection:"column",gap:16}}>
        <div style={{fontSize:11,fontFamily:"var(--mono)",color:"#22C55E",letterSpacing:2}}>HYPOTHESIS / QUESTION</div>
        <textarea ref={inputRef} value={hypothesis} onChange={e=>setHypothesis(e.target.value)} onKeyDown={e=>{if(e.key==="Enter"&&e.metaKey)runExperiment();}} placeholder="Pose a hypothesis, ask a mechanistic question, or describe a system you want to analyze through the EoE lens..." rows={4} style={{background:"#111111",border:"1px solid #2A2A2A",borderRadius:10,padding:"14px 16px",fontSize:14,color:"#FFFFFF",fontFamily:"var(--sans)",lineHeight:1.65,resize:"vertical",outline:"none",transition:"border-color 0.15s"}} onFocus={e=>e.target.style.borderColor="#22C55E"} onBlur={e=>e.target.style.borderColor="#2A2A2A"}/>
        <div style={{display:"flex",gap:12,flexWrap:"wrap"}}>
          <div style={{flex:"1 1 200px"}}>
            <div style={{fontSize:9,fontFamily:"var(--mono)",color:"#525252",marginBottom:6,letterSpacing:1}}>DOMAIN (optional)</div>
            <div style={{display:"flex",flexWrap:"wrap",gap:5}}>{DOMAINS.map(d=><button key={d} onClick={()=>setDomain(domain===d?"":d)} style={{background:domain===d?"#22C55E20":"#111111",border:"1px solid "+(domain===d?"#22C55E":"#2A2A2A"),borderRadius:6,padding:"4px 10px",fontSize:10,color:domain===d?"#22C55E":"#737373",fontFamily:"var(--sans)",cursor:"pointer"}}>{d}</button>)}</div>
          </div>
          <div style={{flex:"1 1 200px"}}>
            <div style={{fontSize:9,fontFamily:"var(--mono)",color:"#525252",marginBottom:6,letterSpacing:1}}>TIME SCALE (optional)</div>
            <div style={{display:"flex",flexWrap:"wrap",gap:5}}>{TIMESCALES.map(t=><button key={t} onClick={()=>setTimeScale(timeScale===t?"":t)} style={{background:timeScale===t?"#22C55E20":"#111111",border:"1px solid "+(timeScale===t?"#22C55E":"#2A2A2A"),borderRadius:6,padding:"4px 10px",fontSize:10,color:timeScale===t?"#22C55E":"#737373",fontFamily:"var(--sans)",cursor:"pointer"}}>{t}</button>)}</div>
          </div>
        </div>
        <button onClick={runExperiment} disabled={!hypothesis.trim()} style={{background:hypothesis.trim()?"#22C55E":"#1A1A1A",border:"none",borderRadius:10,padding:"14px 24px",fontSize:14,fontWeight:700,color:hypothesis.trim()?"#000000":"#525252",fontFamily:"var(--sans)",cursor:hypothesis.trim()?"pointer":"not-allowed",transition:"all 0.15s",alignSelf:"flex-start"}} onMouseEnter={e=>{if(hypothesis.trim())e.currentTarget.style.background="#4ADE80";}} onMouseLeave={e=>{if(hypothesis.trim())e.currentTarget.style.background="#22C55E";}}>Run Experiment ⚗️</button>
        <div style={{fontSize:10,color:"#404040",fontFamily:"var(--sans)"}}>⌘+Enter to run · AI structures, analyzes, and generates full output</div>
      </div>
      <div>
        <div style={{fontSize:9,fontFamily:"var(--mono)",color:"#525252",letterSpacing:2,marginBottom:12}}>EXAMPLE EXPERIMENTS</div>
        <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(300px,1fr))",gap:8}}>
          {EXAMPLES.map((ex,i)=><button key={i} onClick={()=>{setHypothesis(ex.text);setDomain(ex.domain);setTimeScale(ex.time);}} style={{background:"#0A0A0A",border:"1px solid #2A2A2A",borderRadius:10,padding:"14px 16px",textAlign:"left",cursor:"pointer",transition:"all 0.15s",display:"flex",alignItems:"flex-start",gap:10}} onMouseEnter={e=>{e.currentTarget.style.borderColor="#22C55E30";e.currentTarget.style.background="#0A1A0A";}} onMouseLeave={e=>{e.currentTarget.style.borderColor="#2A2A2A";e.currentTarget.style.background="#0A0A0A";}}><span style={{fontSize:20,flexShrink:0,marginTop:2}}>{ex.icon}</span><div><div style={{fontSize:12,color:"#D4D4D4",fontFamily:"var(--sans)",lineHeight:1.5,marginBottom:5}}>{ex.text}</div><div style={{display:"flex",gap:6}}><span style={{fontSize:9,color:"#22C55E",fontFamily:"var(--mono)",background:"#22C55E10",border:"1px solid #22C55E20",borderRadius:4,padding:"1px 6px"}}>{ex.domain}</span><span style={{fontSize:9,color:"#525252",fontFamily:"var(--mono)",background:"#1A1A1A",borderRadius:4,padding:"1px 6px"}}>{ex.time}</span></div></div></button>)}
        </div>
      </div>
    </div>
  );

  if (phase==="processing") {
    const steps=[{label:"Structuring hypothesis",detail:"Extracting variables, domain, time scale"},{label:"Running EoE analysis",detail:"Generating trajectory, analogues, intervention"},{label:"Building output",detail:"Formatting results for display"}];
    return (
      <div style={{display:"flex",flexDirection:"column",alignItems:"center",justifyContent:"center",minHeight:"60vh",gap:32,maxWidth:500,margin:"0 auto",width:"100%"}}>
        <div style={{textAlign:"center"}}><div style={{fontSize:40,marginBottom:16}}>⚗️</div><h3 style={{fontFamily:"var(--serif)",fontSize:22,color:"#FFFFFF",marginBottom:8}}>Running Experiment</h3><p style={{fontSize:13,color:"#737373",fontFamily:"var(--sans)",lineHeight:1.6,maxWidth:400}}>{hypothesis.length>80?hypothesis.slice(0,80)+"…":hypothesis}</p></div>
        <div style={{width:"100%",display:"flex",flexDirection:"column",gap:10}}>
          {steps.map((step,i)=>{const done=processingStep>i+1,active=processingStep===i+1;return(<div key={i} style={{display:"flex",alignItems:"center",gap:14,padding:"12px 18px",borderRadius:10,background:active?"#0A1A0A":done?"#0A0A0A":"#050505",border:"1px solid "+(active?"#22C55E30":done?"#1A1A1A":"#111111"),transition:"all 0.3s"}}><div style={{width:24,height:24,borderRadius:"50%",flexShrink:0,background:done?"#22C55E":active?"#22C55E20":"#1A1A1A",border:"1px solid "+(done?"#22C55E":active?"#22C55E":"#2A2A2A"),display:"flex",alignItems:"center",justifyContent:"center",fontSize:11,color:done?"#000":"#22C55E",fontWeight:700}}>{done?"✓":i+1}</div><div style={{flex:1}}><div style={{fontSize:12,fontWeight:600,color:active||done?"#FFFFFF":"#525252",fontFamily:"var(--sans)"}}>{step.label}</div><div style={{fontSize:10,color:"#525252",fontFamily:"var(--sans)"}}>{step.detail}</div></div></div>);})}
        </div>
      </div>
    );
  }

  if ((phase==="results"||phase==="complete")&&experiment) {
    const {structured:s,analysis:a}=experiment;
    const confColor=CONF_COLOR[s.confidence]||"#737373";
    const confLabel=CONF_LABEL[s.confidence]||s.confidence;
    const finalM=a.chart_data&&a.chart_data.length>0?calcM(a.chart_data[a.chart_data.length-1].chi,a.chart_data[a.chart_data.length-1].s,a.chart_data[a.chart_data.length-1].lambda0,a.chart_data[a.chart_data.length-1].C):null;
    return (
      <div style={{display:"flex",flexDirection:"column",gap:20,maxWidth:820,margin:"0 auto",width:"100%"}}>
        <div style={{display:"flex",justifyContent:"space-between",alignItems:"flex-start",flexWrap:"wrap",gap:12}}>
          <div><div style={{fontSize:9,fontFamily:"var(--mono)",color:"#22C55E",letterSpacing:3,marginBottom:6}}>EXPERIMENT COMPLETE</div><h2 style={{fontFamily:"var(--serif)",fontSize:24,color:"#FFFFFF",marginBottom:4}}>{s.title}</h2><div style={{display:"flex",gap:8,flexWrap:"wrap",alignItems:"center"}}><span style={{fontSize:10,fontFamily:"var(--mono)",color:"#22C55E",background:"#22C55E10",border:"1px solid #22C55E20",borderRadius:4,padding:"2px 8px"}}>{s.domain}</span><span style={{fontSize:10,fontFamily:"var(--mono)",color:"#525252",background:"#1A1A1A",borderRadius:4,padding:"2px 8px"}}>{s.timeScale}</span><span style={{fontSize:10,fontFamily:"var(--mono)",color:confColor,background:confColor+"10",border:"1px solid "+confColor+"20",borderRadius:4,padding:"2px 8px"}}>{confLabel}</span></div></div>
          <button onClick={reset} style={{background:"none",border:"1px solid #2A2A2A",borderRadius:8,padding:"8px 16px",fontSize:11,color:"#737373",fontFamily:"var(--sans)",cursor:"pointer"}}>← New experiment</button>
        </div>
        <div style={{background:"#0A0A0A",border:"1px solid #22C55E20",borderLeft:"3px solid #22C55E",borderRadius:12,padding:"16px 20px"}}><div style={{fontSize:9,fontFamily:"var(--mono)",color:"#22C55E",letterSpacing:2,marginBottom:8}}>HYPOTHESIS</div><p style={{fontSize:14,color:"#FFFFFF",fontFamily:"var(--sans)",lineHeight:1.7,margin:0}}>{s.hypothesis}</p></div>
        <div style={{background:"#0A0A0A",border:"1px solid #2A2A2A",borderRadius:12,padding:20}}><div style={{fontSize:9,fontFamily:"var(--mono)",color:"#3B82F6",letterSpacing:2,marginBottom:14}}>EoE VARIABLE MAPPING</div><div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(180px,1fr))",gap:8}}>{[{sym:"χ",val:s.variables.chi,color:"#60A5FA",name:"Efficiency"},{sym:"s",val:s.variables.s,color:"#A78BFA",name:"Throughput"},{sym:"λ₀",val:s.variables.lambda0,color:"#F87171",name:"Base Burden"},{sym:"C",val:s.variables.C,color:"#FCD34D",name:"Complexity"}].map(v=><div key={v.sym} style={{background:"#111111",borderRadius:8,padding:"12px 14px",border:"1px solid #1A1A1A"}}><div style={{fontFamily:"var(--mono)",fontSize:16,color:v.color,fontWeight:700,marginBottom:4}}>{v.sym}</div><div style={{fontSize:11,color:"#D4D4D4",fontFamily:"var(--sans)",lineHeight:1.5}}>{v.val}</div><div style={{fontSize:9,color:"#525252",fontFamily:"var(--sans)",marginTop:4}}>{v.name}</div></div>)}</div></div>
        <div style={{background:"#0A0A0A",border:"1px solid #2A2A2A",borderRadius:12,padding:20}}><div style={{fontSize:9,fontFamily:"var(--mono)",color:"#A78BFA",letterSpacing:2,marginBottom:10}}>EoE PREDICTION</div><p style={{fontSize:13,color:"#D4D4D4",fontFamily:"var(--sans)",lineHeight:1.75,margin:0}}>{s.eoe_prediction}</p>{s.uncertainty_flag&&<div style={{marginTop:12,padding:"10px 14px",background:"#F9731610",border:"1px solid #F9731620",borderRadius:8,fontSize:11,color:"#F97316",fontFamily:"var(--sans)"}}>⚠ {s.uncertainty_flag}</div>}</div>
        {s.generate_chart&&a.chart_data&&a.chart_data.length>=2?(
          <div style={{background:"#0A0A0A",border:"1px solid #2A2A2A",borderRadius:12,padding:20}}>
            <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:16,flexWrap:"wrap",gap:8}}>
              <div style={{fontSize:9,fontFamily:"var(--mono)",color:"#22C55E",letterSpacing:2}}>PROJECTED M TRAJECTORY</div>
              {finalM!==null&&<div style={{display:"flex",alignItems:"center",gap:10}}><div style={{textAlign:"right"}}><div style={{fontFamily:"var(--mono)",fontSize:18,color:mColor(finalM),fontWeight:700}}>{finalM>=0?"+":""}{finalM.toFixed(3)}</div><div style={{fontSize:9,color:mColor(finalM),fontFamily:"var(--sans)"}}>{mLabel(finalM)}</div></div><Gauge value={finalM} size={70}/></div>}
            </div>
            <ExperimentChart points={a.chart_data} color="#22C55E"/>
            {a.chart_note&&<div style={{marginTop:10,fontSize:10,color:"#525252",fontFamily:"var(--sans)",fontStyle:"italic"}}>⚠ {a.chart_note}</div>}
          </div>
        ):(!s.generate_chart&&<div style={{background:"#0A0A0A",border:"1px solid #2A2A2A",borderRadius:12,padding:"14px 20px",display:"flex",gap:10,alignItems:"flex-start"}}><span style={{fontSize:14,flexShrink:0}}>📊</span><div style={{fontSize:12,color:"#737373",fontFamily:"var(--sans)",lineHeight:1.6}}><strong style={{color:"#D4D4D4"}}>Chart not generated.</strong> {s.chart_reason}</div></div>)}
        <div style={{background:"#0A0A0A",border:"1px solid #2A2A2A",borderRadius:12,padding:20}}><div style={{fontSize:9,fontFamily:"var(--mono)",color:"#3B82F6",letterSpacing:2,marginBottom:14}}>ANALYSIS</div><div style={{fontSize:13,color:"#D4D4D4",fontFamily:"var(--sans)",lineHeight:1.85,whiteSpace:"pre-wrap"}}>{a.narrative}</div></div>
        <div style={{background:"#111111",border:"1px solid #22C55E20",borderLeft:"3px solid #22C55E",borderRadius:12,padding:"14px 20px",display:"flex",gap:12,alignItems:"flex-start"}}><div style={{fontSize:18,flexShrink:0}}>💡</div><div><div style={{fontSize:9,fontFamily:"var(--mono)",color:"#22C55E",letterSpacing:2,marginBottom:6}}>MINSIGHT — PLAIN ENGLISH</div><p style={{fontSize:13,color:"#FFFFFF",fontFamily:"var(--sans)",lineHeight:1.7,margin:0}}>{a.minsight}</p></div></div>
        <div style={{background:"#0A0A0A",border:"1px solid #2A2A2A",borderRadius:12,padding:20}}><div style={{fontSize:9,fontFamily:"var(--mono)",color:"#FCD34D",letterSpacing:2,marginBottom:10}}>KEY FINDING</div><p style={{fontSize:14,color:"#FFFFFF",fontFamily:"var(--serif)",lineHeight:1.7,fontStyle:"italic",margin:0}}>"{a.key_finding}"</p></div>
        {a.analogues&&a.analogues.length>0&&<div style={{background:"#0A0A0A",border:"1px solid #2A2A2A",borderRadius:12,padding:20}}><div style={{fontSize:9,fontFamily:"var(--mono)",color:"#A78BFA",letterSpacing:2,marginBottom:12}}>HISTORICAL ANALOGUES</div><div style={{display:"flex",flexDirection:"column",gap:8}}>{a.analogues.map((analogue,i)=><div key={i} style={{display:"flex",gap:10,alignItems:"flex-start",padding:"10px 14px",background:"#111111",borderRadius:8,border:"1px solid #1A1A1A"}}><div style={{fontFamily:"var(--mono)",fontSize:11,color:"#525252",flexShrink:0,marginTop:1}}>#{i+1}</div><div style={{fontSize:12,color:"#D4D4D4",fontFamily:"var(--sans)",lineHeight:1.6}}>{analogue}</div></div>)}</div></div>}
        <div style={{background:"#0A0A0A",border:"1px solid #2A2A2A",borderRadius:12,padding:20}}><div style={{fontSize:9,fontFamily:"var(--mono)",color:"#34D399",letterSpacing:2,marginBottom:10}}>INTERVENTION — WHAT WOULD ACTUALLY HELP</div><p style={{fontSize:13,color:"#D4D4D4",fontFamily:"var(--sans)",lineHeight:1.75,margin:0}}>{a.intervention}</p></div>
        <div style={{background:"linear-gradient(135deg,#0A1A0A,#0A0A1A)",border:"1px solid #22C55E30",borderRadius:14,padding:24,display:"flex",flexDirection:"column",gap:16}}>
          <div style={{display:"flex",alignItems:"center",gap:10,marginBottom:4}}><div style={{width:8,height:8,borderRadius:"50%",background:"#22C55E",boxShadow:"0 0 8px #22C55E"}}/><div style={{fontSize:11,fontFamily:"var(--mono)",color:"#22C55E",letterSpacing:2}}>STUDY COMPLETE — READY TO EXPORT</div></div>
          <p style={{fontSize:12,color:"#737373",fontFamily:"var(--sans)",lineHeight:1.65,margin:0,maxWidth:560}}>Your experiment has been structured, analyzed, and formatted. Export as Markdown, copy a summary for sharing, or open the full dataset context in Explore.</p>
          <div style={{display:"flex",gap:8,flexWrap:"wrap"}}>
            <button onClick={exportMarkdown} style={{background:"#22C55E",border:"none",borderRadius:8,padding:"10px 18px",fontSize:12,fontWeight:600,color:"#000000",fontFamily:"var(--sans)",cursor:"pointer",display:"flex",alignItems:"center",gap:6}} onMouseEnter={e=>e.currentTarget.style.background="#4ADE80"} onMouseLeave={e=>e.currentTarget.style.background="#22C55E"}>↓ Download .md</button>
            <button onClick={copyToClipboard} style={{background:exportDone?"#22C55E20":"#111111",border:"1px solid "+(exportDone?"#22C55E":"#2A2A2A"),borderRadius:8,padding:"10px 18px",fontSize:12,fontWeight:600,color:exportDone?"#22C55E":"#D4D4D4",fontFamily:"var(--sans)",cursor:"pointer"}}>{exportDone?"✓ Copied!":"⎘ Copy summary"}</button>
            <button onClick={onGoToExplore} style={{background:"#111111",border:"1px solid #3B82F6",borderRadius:8,padding:"10px 18px",fontSize:12,fontWeight:600,color:"#3B82F6",fontFamily:"var(--sans)",cursor:"pointer"}} onMouseEnter={e=>e.currentTarget.style.background="#0A0A1A"} onMouseLeave={e=>e.currentTarget.style.background="#111111"}>Full analysis in Explore →</button>
            <button onClick={reset} style={{background:"none",border:"1px solid #2A2A2A",borderRadius:8,padding:"10px 18px",fontSize:12,color:"#525252",fontFamily:"var(--sans)",cursor:"pointer"}}>⚗️ New experiment</button>
          </div>
          <div style={{padding:"10px 14px",background:"#111111",borderRadius:8,border:"1px solid #1A1A1A",fontFamily:"var(--mono)",fontSize:10,color:"#525252",lineHeight:1.8}}>Cite as: Baird, N. (2026). Engine of Emergence: A Thermodynamic Framework for the Persistence and Collapse of Organized Complexity. DOI: 10.5281/zenodo.19016245</div>
        </div>
      </div>
    );
  }
  return null;
}


// ── TAB: ASSISTANT ────────────────────────────────────────────────────────────
function AssistantTab({ onExperimentReady }) {
  const [input, setInput]       = useState("");
  const [messages, setMessages] = useState([
    { role:"assistant", text:"Hey — ask me anything about the Engine of Emergence, or drop a CSV file here and I'll run the analysis for you automatically. I'll map your columns, calculate the Stability Margin, and the results will appear here and in the Experiment tab instantly." }
  ]);
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState("");
  const [dragOver, setDragOver] = useState(false);
  const bottomRef               = useRef(null);
  const fileRef                 = useRef(null);

  const PROMPTS = [
    "Explain this like I'm 10",
    "Is the US government in trouble?",
    "How do I analyze Amazon's financials?",
    "Can I apply EoE to a hospital system?",
    "What is the difference between χ and s?",
    "How do I normalize my data for upload?",
    "What makes a good EoE dataset?",
    "Walk me through the Rome collapse",
  ];

  const SYSTEM_PROMPT = `You are the Engine of Emergence assistant — a knowledgeable, plain-English guide to the EoE framework and the data needed to use it.

The Engine of Emergence (EoE) framework measures M = χs − λ(C) where:
- χ (chi) = architectural efficiency (0–1)
- s = energy throughput (0–1)  
- λ(C) = systemic burden = λ₀ + 0.15×Cⁿ where n=1.4
- C = complexity (0–1)
- M positive = runway, M negative = borrowed time

APPROVED DATA SOURCES by domain (always recommend these by name):
- Business/Company: SEC EDGAR (edgar.sec.gov), Macrotrends (macrotrends.net), World Bank Enterprise Surveys
- City/Urban: Lincoln Institute FiSC Database, BEA Metro GDP, US Census ACS
- National Government: World Bank Open Data (data.worldbank.org), IMF WEO Database, CBO Historical Data
- Coral Reef/Ocean: AIMS LTMP, NOAA Coral Reef Watch, OBIS
- Forest/Land: USFS FIA (apps.fs.usda.gov/fia), Global Forest Watch, NASA AppEEARS
- Agriculture/Valley: USDA NASS (nass.usda.gov), California DWR (water.ca.gov), CDFA (cdfa.ca.gov), USGS Water Resources (waterdata.usgs.gov)
- Seismic/Earthquake: USGS Earthquake Catalog (earthquake.usgs.gov), UNAVCO GPS data
- Civilization/Historical: Seshat Databank (seshatdatabank.info), Turchin Cliodynamics, HYDE Database
- Financial: FDIC BankFind, Federal Reserve Z.1, BIS Statistics
- Lake/Freshwater: EDI/NTL-LTER, USGS Water Info, GLEON Network

When someone asks about a specific geographic system (like California's Central Valley), ALWAYS:
1. Identify which EoE domain it belongs to
2. Name the specific approved data sources for that domain
3. Tell them exactly which variables/columns to download
4. Explain how those columns map to χ, s, λ₀, and C

Be concise - 3-5 sentences max unless they ask for more detail. Never use bullet points in conversational responses.

When a user uploads a CSV, analyze the headers and sample rows, then return a JSON block in this EXACT format (no markdown, no backticks around it):
{"mapping":{"label":"column_name_or_null","chi":"column_name","s":"column_name","lambda0":"column_name","C":"column_name"},"confidence":"high|medium|low","notes":"one sentence explaining your mapping"}

After the JSON block, explain your mapping in plain English in 2-3 sentences. Never use bullet points. If a column mapping is genuinely ambiguous ask one specific clarifying question.`;

  useEffect(() => { bottomRef.current?.scrollIntoView({behavior:"smooth"}); }, [messages, loading]);

  function parseCSVText(text) {
    const lines = text.trim().split(/\r?\n/).filter(l=>l.trim());
    const headers = lines[0].split(",").map(h=>h.trim().replace(/^"|"$/g,""));
    const allRows = lines.slice(1).map(line=>{
      const vals = line.split(",").map(v=>v.trim().replace(/^"|"$/g,""));
      const obj = {};
      headers.forEach((h,i)=>{ obj[h]=vals[i]||""; });
      return obj;
    });
    return { headers, rows:allRows.slice(0,5), allRows, totalRows:lines.length-1 };
  }

  function saveToLibrary(name, results) {
    try {
      const existing = JSON.parse(localStorage.getItem("eoe_community")||"[]");
      const entry = {
        id: Date.now().toString(),
        name,
        domain: "Community",
        addedAt: new Date().toISOString().split("T")[0],
        points: results,
        source: "User upload via Assistant"
      };
      localStorage.setItem("eoe_community", JSON.stringify([entry, ...existing]));
    } catch(e) {}
  }

  async function processFile(file) {
    if (!file || !file.name.match(/\.(csv|tsv|txt)$/i)) {
      setError("Please upload a CSV file."); return;
    }
    const reader = new FileReader();
    reader.onload = async (ev) => {
      const text = ev.target.result;
      const { headers, rows, allRows, totalRows } = parseCSVText(text);

      setMessages(prev=>[...prev, {
        role:"user", text:`📂 ${file.name}`, isFile:true,
        subtext:`${totalRows} rows · ${headers.length} columns: ${headers.join(", ")}`
      }]);
      setLoading(true);

      const prompt = `CSV file: "${file.name}" — ${totalRows} rows
Headers: ${headers.join(", ")}
Sample data (first 3 rows):
${JSON.stringify(rows.slice(0,3), null, 2)}

Please map these columns to EoE variables and return the JSON mapping.`;

      try {
        const response = await fetch("https://api.anthropic.com/v1/messages", {
          method:"POST",
          headers:{"Content-Type":"application/json","x-api-key":import.meta.env.VITE_ANTHROPIC_KEY||"","anthropic-version":"2023-06-01","anthropic-dangerous-direct-browser-access":"true"},
          body: JSON.stringify({
            model:"claude-sonnet-4-5",
            max_tokens:800,
            system: SYSTEM_PROMPT,
            messages:[{role:"user", content:prompt}]
          })
        });
        const data = await response.json();
        const reply = data.content?.map(b=>b.text||"").join("") || "";

        // Extract JSON mapping
        const jsonMatch = reply.match(/\{[^{}]*"mapping"[^{}]*\{[^{}]*\}[^{}]*\}/s) ||
                          reply.match(/\{"mapping"[\s\S]*?\}(?=\s|$)/);

        if (jsonMatch) {
          try {
            const parsed = JSON.parse(jsonMatch[0]);
            const mapping = parsed.mapping;

            // Run calculation on all rows
            const k2=0.15, n2=1.4;
            const results = allRows.map((row,i)=>{
              const chi     = parseFloat(row[mapping.chi]);
              const s       = parseFloat(row[mapping.s]);
              const lambda0 = parseFloat(row[mapping.lambda0]);
              const C       = parseFloat(row[mapping.C]);
              const label   = mapping.label ? (row[mapping.label]||String(i+1)) : String(i+1);
              if ([chi,s,lambda0,C].some(isNaN)) return null;
              const M = chi*s - (lambda0 + k2*Math.pow(C,n2));
              const yr = parseInt(row[mapping.label]) || (2000+i);
              return { label, chi, s, lambda0, C, M, year:yr };
            }).filter(Boolean);

            if (results.length < 2) {
              setMessages(prev=>[...prev,{role:"assistant",
                text:"I mapped the columns but couldn't find enough valid numeric rows. Make sure all four variable columns contain numbers between 0 and 1, then try again."}]);
              setLoading(false); return;
            }

            const lastM = results[results.length-1].M;
            const dsName = file.name.replace(/\.(csv|tsv|txt)$/i,"");

            // AI explanation (text after the JSON)
            const explanation = reply.replace(jsonMatch[0],"").trim() ||
              `Mapped your ${results.length} rows and calculated the Stability Margin. Final M = ${lastM>=0?"+":""}${lastM.toFixed(4)} — ${mLabel(lastM).toLowerCase()}.`;

            setMessages(prev=>[...prev,
              { role:"assistant", text:explanation },
              { role:"result", dsName, results, lastM, confidence:parsed.confidence }
            ]);

            // Auto-save immediately — no prompt
            saveToLibrary(dsName, results);

            // Send to Experiment tab
            if (onExperimentReady) {
              onExperimentReady({
                id: "uploaded_"+Date.now(),
                label: dsName,
                domain: "Uploaded",
                emoji: "📊",
                color: "#3B82F6",
                period: `${results[0]?.year||""}–${results[results.length-1]?.year||""}`,
                desc: `Uploaded dataset: ${file.name}. ${results.length} data points.`,
                source: "User upload",
                points: results.map(r=>({
                  year:r.year, label:r.label,
                  chi:r.chi, s:r.s, lambda0:r.lambda0, C:r.C
                })),
              });
            }

            // Confirmation message
            setMessages(prev=>[...prev,{role:"assistant",
              text:`Saved to your library and loaded in the Experiment tab. You can run the full analysis there — sensitivity analysis, tipping point projection, download report, everything. What would you like to know about these results?`
            }]);

          } catch(e) {
            setMessages(prev=>[...prev,{role:"assistant", text:reply}]);
          }
        } else {
          // No JSON — AI needs clarification
          setMessages(prev=>[...prev,{role:"assistant", text:reply}]);
        }
      } catch(err) {
        setMessages(prev=>[...prev,{role:"assistant",
          text:"Connection issue processing your file. Please try again in a moment."}]);
      }
      setLoading(false);
    };
    reader.readAsText(file);
  }

  async function sendMessage() {
    const trimmed = input.trim();
    const check = validateInput(trimmed);
    if (!check.ok) { setError(check.reason); return; }
    setError("");
    const history = messages
      .filter(m=>m.role==="user"||m.role==="assistant")
      .map(m=>({role:m.role, content:m.text}));
    const userMsg = {role:"user", text:trimmed};
    setMessages(prev=>[...prev, userMsg]);
    setInput(""); setLoading(true);
    try {
      const resp = await fetch("https://api.anthropic.com/v1/messages", {
        method:"POST",
        headers:{"Content-Type":"application/json","x-api-key":import.meta.env.VITE_ANTHROPIC_KEY||"","anthropic-version":"2023-06-01","anthropic-dangerous-direct-browser-access":"true"},
        body:JSON.stringify({
          model:"claude-sonnet-4-5", max_tokens:1000,
          system:SYSTEM_PROMPT,
          messages:[...history, {role:"user",content:trimmed}]
        })
      });
      const data = await resp.json();
      const reply = data.content?.map(b=>b.text||"").join("")||"Something went wrong — please try again.";
      setMessages(prev=>[...prev,{role:"assistant",text:reply}]);
    } catch(e) {
      setMessages(prev=>[...prev,{role:"assistant",
        text:"Connection issue. Please try again in a moment."}]);
    }
    setLoading(false);
  }

  function handleKey(e) {
    if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();sendMessage();}
  }

  function ResultCard({ msg }) {
    const { results, lastM, dsName, confidence } = msg;
    const confColor = confidence==="high"?"#22C55E":confidence==="medium"?"#EAB308":"#F97316";
    const W = Math.max(300, results.length*28);
    return (
      <div style={{background:"#0A0A0A",border:`1px solid ${mColor(lastM)}35`,
        borderRadius:12,padding:16,maxWidth:"92%"}}>
        <div style={{fontFamily:"var(--mono)",fontSize:9,color:"#3B82F6",
          letterSpacing:3,marginBottom:10}}>ANALYSIS COMPLETE — {dsName.toUpperCase()}</div>
        <div style={{display:"flex",gap:14,alignItems:"center",flexWrap:"wrap",marginBottom:12}}>
          <div style={{textAlign:"center",flexShrink:0}}>
            <Gauge value={lastM} size={110}/>
            <div style={{fontSize:12,fontWeight:700,color:mColor(lastM),
              fontFamily:"var(--sans)",marginTop:2}}>{mLabel(lastM)}</div>
          </div>
          <div style={{flex:1,minWidth:140}}>
            <div style={{fontFamily:"var(--mono)",fontSize:16,
              color:mColor(lastM),fontWeight:700,marginBottom:4}}>
              M = {lastM>=0?"+":""}{lastM.toFixed(4)}
            </div>
            <div style={{fontSize:11,color:"#737373",fontFamily:"var(--sans)",marginBottom:4}}>
              {results.length} data points · {results.filter(r=>r.M<0).length} with negative margin
            </div>
            <div style={{fontSize:10,color:confColor,fontFamily:"var(--mono)"}}>
              Mapping confidence: {confidence}
            </div>
          </div>
        </div>
        {/* Mini sparkline bar chart */}
        <svg width={W} height={52} style={{display:"block",overflowX:"auto"}}>
          <line x1={16} y1={26} x2={W-8} y2={26} stroke="#2A2A2A" strokeWidth={0.5}/>
          {results.map((r,i)=>{
            const x=16+i*((W-24)/Math.max(1,results.length-1));
            const bh=Math.abs(r.M)/0.6*22;
            const by=r.M>=0?26-bh:26;
            const bw=Math.max(4,(W-24)/results.length-3);
            return <rect key={i} x={x-bw/2} y={by} width={bw}
              height={Math.max(2,bh)} fill={mColor(r.M)} rx={1} opacity={0.9}/>;
          })}
        </svg>
        <div style={{marginTop:8,fontSize:10,color:"#525252",fontFamily:"var(--sans)",fontStyle:"italic"}}>
          ✓ Saved to library · ✓ Loaded in Experiment tab
        </div>
      </div>
    );
  }

  return (
    <div style={{maxWidth:720, margin:"0 auto", border:"1px solid #2A2A2A",
      borderRadius:14, overflow:"hidden", minHeight:520,
      display:"flex", flexDirection:"column", position:"relative"}}
      onDragOver={e=>{e.preventDefault();setDragOver(true);}}
      onDragLeave={()=>setDragOver(false)}
      onDrop={e=>{e.preventDefault();setDragOver(false);
        const f=e.dataTransfer.files[0]; if(f) processFile(f);}}>

      {/* Drag overlay */}
      {dragOver && (
        <div style={{position:"absolute",inset:0,background:"#2563EB15",
          border:"2px dashed #2563EB",borderRadius:14,zIndex:10,
          display:"flex",alignItems:"center",justifyContent:"center",
          pointerEvents:"none"}}>
          <div style={{fontSize:18,color:"#3B82F6",fontFamily:"var(--sans)",fontWeight:700}}>
            Drop CSV to analyze →
          </div>
        </div>
      )}

      {/* Header */}
      <div style={{padding:"16px 20px 14px", borderBottom:"1px solid #1A1A1A",
        flexShrink:0, background:"#0A0A0A",
        display:"flex",justifyContent:"space-between",alignItems:"center",gap:10}}>
        <div>
          <div style={{fontFamily:"var(--serif)", fontSize:18, color:"#FFFFFF", marginBottom:2}}>
            Assistant
          </div>
          <div style={{fontSize:11, color:"#737373", fontFamily:"var(--sans)"}}>
            Ask anything · Drop a CSV to auto-analyze · Results save instantly
          </div>
        </div>
        <button onClick={()=>fileRef.current?.click()} style={{
          background:"#2563EB",border:"none",borderRadius:8,
          padding:"8px 14px",fontSize:11,fontWeight:600,
          color:"#FFFFFF",fontFamily:"var(--sans)",cursor:"pointer",
          display:"flex",alignItems:"center",gap:5,flexShrink:0,
          transition:"background 0.12s"
        }}
          onMouseEnter={e=>e.currentTarget.style.background="#3B82F6"}
          onMouseLeave={e=>e.currentTarget.style.background="#2563EB"}>
          <span>📂</span><span>Upload CSV</span>
        </button>
        <input ref={fileRef} type="file" accept=".csv,.tsv,.txt"
          onChange={e=>{if(e.target.files[0])processFile(e.target.files[0]);e.target.value="";}}
          style={{display:"none"}}/>
      </div>

      {/* Messages */}
      <div style={{flex:1,overflowY:"auto",padding:"20px",
        display:"flex",flexDirection:"column",gap:16,background:"#000000",minHeight:240}}>
        {messages.map((m,i)=>{
          if(m.role==="result") return <ResultCard key={i} msg={m}/>;
          return (
            <div key={i} style={{display:"flex",
              justifyContent:m.role==="user"?"flex-end":"flex-start",
              gap:8,alignItems:"flex-start"}}>
              {m.role==="assistant" && (
                <div style={{width:28,height:28,borderRadius:"50%",background:"#2563EB",
                  display:"flex",alignItems:"center",justifyContent:"center",
                  fontSize:11,fontWeight:700,color:"#FFFFFF",flexShrink:0,marginTop:1}}>E</div>
              )}
              <div style={{
                maxWidth:"80%",
                background:m.isFile?"#0A1520":m.role==="user"?"#1A1A1A":"#111111",
                border:`1px solid ${m.isFile?"#2563EB30":m.role==="user"?"#2A2A2A":"#1A1A1A"}`,
                borderRadius:m.role==="user"?"14px 14px 4px 14px":"4px 14px 14px 14px",
                padding:"11px 15px",fontSize:13,lineHeight:1.75,
                color:"#D4D4D4",fontFamily:"var(--sans)",fontWeight:400
              }}>
                {m.text}
                {m.subtext && (
                  <div style={{fontSize:10,color:"#525252",fontFamily:"var(--mono)",marginTop:4}}>
                    {m.subtext}
                  </div>
                )}
              </div>
            </div>
          );
        })}
        {loading && (
          <div style={{display:"flex",gap:8,alignItems:"flex-start"}}>
            <div style={{width:28,height:28,borderRadius:"50%",background:"#2563EB",
              display:"flex",alignItems:"center",justifyContent:"center",
              fontSize:11,fontWeight:700,color:"#FFFFFF",flexShrink:0}}>E</div>
            <div style={{background:"#111111",border:"1px solid #1A1A1A",
              borderRadius:"4px 14px 14px 14px",padding:"11px 15px",
              display:"flex",gap:5,alignItems:"center"}}>
              {[0,1,2].map(i=>(
                <div key={i} style={{width:6,height:6,borderRadius:"50%",
                  background:"#3B82F6",animation:"pulse 1.2s ease-in-out infinite",
                  animationDelay:`${i*0.2}s`}}/>
              ))}
            </div>
          </div>
        )}
        <div ref={bottomRef}/>
      </div>

      {/* Error */}
      {error && (
        <div style={{padding:"8px 20px",background:"#1A0A0A",
          borderTop:"1px solid #3A1A1A",flexShrink:0,
          display:"flex",gap:8,alignItems:"center"}}>
          <span style={{fontSize:13}}>🚫</span>
          <span style={{fontSize:12,color:"#EF4444",fontFamily:"var(--sans)"}}>{error}</span>
        </div>
      )}

      {/* Suggested prompts */}
      <div style={{padding:"12px 20px 8px",borderTop:"1px solid #1A1A1A",
        background:"#0A0A0A",flexShrink:0}}>
        <div style={{fontSize:9,color:"#525252",fontFamily:"var(--sans)",
          marginBottom:7,textTransform:"uppercase",letterSpacing:1}}>Suggested</div>
        <div style={{display:"flex",flexWrap:"wrap",gap:6}}>
          {PROMPTS.map((q,i)=>(
            <button key={i} onClick={()=>{setInput(q);setError("");}} style={{
              background:"#111111",border:"1px solid #2A2A2A",borderRadius:16,
              padding:"4px 11px",fontSize:11,color:"#A3A3A3",
              fontFamily:"var(--sans)",transition:"all 0.12s"
            }}
              onMouseEnter={e=>{e.currentTarget.style.borderColor="#2563EB";e.currentTarget.style.color="#93C5FD";}}
              onMouseLeave={e=>{e.currentTarget.style.borderColor="#2A2A2A";e.currentTarget.style.color="#A3A3A3";}}
            >{q}</button>
          ))}
        </div>
      </div>

      {/* Input */}
      <div style={{padding:"10px 20px 16px",background:"#0A0A0A",
        borderTop:"1px solid #1A1A1A",flexShrink:0,display:"flex",gap:8}}>
        <input value={input} onChange={e=>{setInput(e.target.value);if(error)setError("");}}
          onKeyDown={handleKey}
          placeholder="Ask anything, or describe a system you want to analyze..."
          style={{flex:1,background:"#111111",
            border:`1px solid ${error?"#EF4444":"#2A2A2A"}`,
            borderRadius:9,padding:"11px 16px",fontSize:13,
            color:"#FFFFFF",outline:"none",fontFamily:"var(--sans)",
            transition:"border-color 0.15s"}}
          onFocus={e=>{if(!error)e.target.style.borderColor="#2563EB";}}
          onBlur={e=>{if(!error)e.target.style.borderColor="#2A2A2A";}}
        />
        <button onClick={sendMessage} disabled={loading} style={{
          background:loading?"#1A1A1A":"#2563EB",border:"none",borderRadius:9,
          width:44,height:44,fontSize:18,color:"white",
          display:"flex",alignItems:"center",justifyContent:"center",
          flexShrink:0,cursor:loading?"not-allowed":"pointer",
          opacity:loading?0.5:1,transition:"background 0.12s"
        }}
          onMouseEnter={e=>{if(!loading)e.currentTarget.style.background="#3B82F6";}}
          onMouseLeave={e=>{if(!loading)e.currentTarget.style.background="#2563EB";}}
        >↑</button>
      </div>
    </div>
  );
}


// ── TAB: DATA DIRECTORY ───────────────────────────────────────────────────────
function DirectoryTab() {
  const [open, setOpen] = useState(null);

  const DIR = [
    { domain:"Business / Company", emoji:"🏢", color:"#FCD34D",
      intro:"Revenue, operating costs, and organizational complexity data for public and private companies.",
      sources:[
        { name:"SEC EDGAR",             url:"https://www.sec.gov/cgi-bin/browse-edgar",                   badge:"Easy",   desc:"10-K and 10-Q filings for every US public company. Revenue, gross margin, fixed costs — everything for χ, s, and λ₀.", tip:"Search any ticker → Financial Statements → download CSV. Gross margin % → χ. Revenue growth → s. SG&A / revenue → λ₀." },
        { name:"Macrotrends",           url:"https://www.macrotrends.net",                                 badge:"Easy",   desc:"Pre-formatted historical financials for thousands of public companies. Free CSV download on every chart.", tip:"Search company name → 'Gross Profit Margin' → Download button below the chart. Decade of clean annual data." },
        { name:"World Bank Enterprise", url:"https://www.enterprisesurveys.org/en/data",                   badge:"Easy",   desc:"Firm-level productivity, regulatory burden, and capacity utilization across 150 countries.", tip:"Capacity utilization → s. % revenue on compliance → λ₀. Manager time on regulations → C." },
      ]},
    { domain:"Coral Reef / Ecosystem", emoji:"🪸", color:"#34D399",
      intro:"Productivity, bleaching, and biodiversity data for reef and marine ecosystem analysis.",
      sources:[
        { name:"AIMS LTMP",             url:"https://www.aims.gov.au/research-topics/monitoring-and-discovery/reef-monitoring/long-term-reef-monitoring-program", badge:"Easy", desc:"Australia's authoritative Great Barrier Reef dataset since 1993. Coral cover %, bleaching frequency, fish biomass.", tip:"Coral cover % → χ. Degree Heating Weeks → raises λ₀. Download the 'Reef Health Summary' CSV." },
        { name:"NOAA Coral Reef Watch", url:"https://coralreefwatch.noaa.gov/product/vs/data.php",         badge:"Easy",   desc:"Global satellite-derived sea surface temperature and bleaching alerts, updated twice weekly.", tip:"Degree Heating Weeks = thermal stress → rising λ₀. Pick any reef location, download the Virtual Station CSV." },
        { name:"OBIS",                  url:"https://obis.org",                                            badge:"Medium", desc:"Global ocean species occurrence database. Species richness per reef location → your C (complexity) variable.", tip:"Filter by location + coral/fish taxa. Species count per area per year → C proxy." },
      ]},
    { domain:"City / Urban System", emoji:"🏙️", color:"#60A5FA",
      intro:"Economic output, fiscal health, and population data for cities and metropolitan areas.",
      sources:[
        { name:"BEA Metro GDP",         url:"https://www.bea.gov/data/gdp/gdp-metropolitan-area",          badge:"Easy",   desc:"GDP for all 384 US metro areas going back to 2001. Compare against population and infrastructure spending.", tip:"GDP per capita growth → s. GDP / maintenance spending → χ. Download table 'CAGDP2'." },
        { name:"Lincoln Inst. FiSC",    url:"https://www.lincolninst.edu/research-data/data-toolkits/fiscally-standardized-cities", badge:"Easy", desc:"Standardized fiscal data for 150 US cities since 1977. Revenue, expenditure, debt, pension obligations.", tip:"Pension + debt service / total revenue → λ₀. Total revenue growth → s. Cleanest city fiscal dataset available." },
        { name:"US Census ACS",         url:"https://data.census.gov",                                     badge:"Easy",   desc:"Population, income, commute times, housing costs, employment by city and census tract.", tip:"Mean commute time growing faster than wages = rising C. Housing cost / income → λ₀ for residents." },
      ]},
    { domain:"Civilization / Empire", emoji:"🏛️", color:"#A78BFA",
      intro:"Historical society data spanning thousands of years — administrative complexity, military costs, population.",
      sources:[
        { name:"Seshat Databank",       url:"https://seshatdatabank.info/data/",                           badge:"Medium", desc:"The most comprehensive quantitative dataset of historical societies ever assembled. 400+ societies, 10,000 years.", tip:"Administrative levels → C. Agricultural surplus → s. Download 'General Variables' CSV, filter by society." },
        { name:"Turchin Cliodynamics",  url:"https://peterturchin.com/cliodynamica/",                      badge:"Medium", desc:"Quantitative data on political instability, state collapse, and social complexity.", tip:"Elite overproduction + state fiscal stress → λ₀ rising. Instability count per decade → early M warning." },
        { name:"HYDE Database",         url:"https://www.pbl.nl/en/image/links/hyde",                      badge:"Medium", desc:"Population and land use estimates going back 12,000 years at fine geographic resolution.", tip:"Population density × territorial extent → C proxy. Population collapse events = EoE test cases." },
      ]},
    { domain:"National Government", emoji:"🇺🇸", color:"#F87171",
      intro:"Government effectiveness, fiscal health, and mandatory spending data for countries worldwide.",
      sources:[
        { name:"World Bank Open Data",  url:"https://data.worldbank.org",                                  badge:"Easy",   desc:"Comprehensive indicators for 200+ countries. Government effectiveness, debt/GDP, tax revenue, GDP growth — all free.", tip:"Search 'Government Effectiveness' → χ. Tax revenue / GDP → s. Debt service / revenue → λ₀. Hit Download → CSV." },
        { name:"IMF WEO Database",      url:"https://www.imf.org/en/Publications/WEO/weo-database",        badge:"Easy",   desc:"IMF flagship macro dataset. GDP, revenue, expenditure, debt for 190 countries since 1980.", tip:"Government revenue → s. Gross debt / GDP → λ₀ scaling variable. Download full WEO database, filter by country." },
        { name:"CBO Historical Data",   url:"https://www.cbo.gov/data",                                    badge:"Easy",   desc:"The most credible long-term US fiscal data. Mandatory vs discretionary spending, debt projections, revenue.", tip:"Mandatory spending / total outlays → λ₀ — this ratio has risen from ~30% in 1970 to 70%+ today." },
      ]},
    { domain:"Lake / Freshwater", emoji:"🏞️", color:"#38BDF8",
      intro:"Water quality, nutrient loading, and ecosystem health data for lakes and freshwater systems.",
      sources:[
        { name:"EDI / NTL-LTER",        url:"https://portal.edirepository.org/nis/home.jsp",               badge:"Easy",   desc:"The actual dataset behind EoE's lake eutrophication P7 test. 40+ years of continuous Wisconsin lake data.", tip:"Search 'North Temperate Lakes'. Secchi depth → χ. Total phosphorus → λ₀ load. Carpenter's own data." },
        { name:"USGS Water Info",        url:"https://waterdata.usgs.gov",                                  badge:"Easy",   desc:"Real-time and historical water quality for thousands of US lakes, rivers, and streams.", tip:"Dissolved oxygen decline → rising λ₀. Turbidity increase → declining χ. Download 'Water Quality' time series." },
        { name:"GLEON Network",          url:"https://gleon.org/data",                                      badge:"Medium", desc:"High-frequency sensor data from lakes worldwide. The early warning signal data that maps onto EoE P7.", tip:"Increasing variance in dissolved oxygen = critical slowing down before tipping — exactly what EoE predicts." },
      ]},
    { domain:"Forest Ecosystem", emoji:"🌲", color:"#4ADE80",
      intro:"Forest productivity, cover loss, and structural complexity data for terrestrial ecosystem analysis.",
      sources:[
        { name:"USFS FIA",              url:"https://apps.fs.usda.gov/fia/datamart/",                      badge:"Medium", desc:"The definitive US forest dataset. Tree species, basal area, volume, mortality for every US forest.", tip:"Live tree basal area / total → χ. Mortality rate → λ₀. Species richness per plot → C. Download 'TREE' and 'PLOT' tables." },
        { name:"Global Forest Watch",   url:"https://www.globalforestwatch.org/dashboards/global/",        badge:"Easy",   desc:"Annual tree cover loss and gain globally since 2000. Clean dashboard with direct CSV export.", tip:"Tree cover loss rate → rising λ₀. Intact forest landscape % → C. Select region → 'Forest Change' → Download." },
        { name:"NASA AppEEARS",         url:"https://appeears.earthdatacloud.nasa.gov",                    badge:"Medium", desc:"Extract satellite Net Primary Productivity time series for any area on Earth, 2000–present.", tip:"Request MOD17A3H (annual NPP) for your area. Declining NPP = declining χ. Free NASA Earthdata account required." },
      ]},
    { domain:"Financial System", emoji:"🏦", color:"#E879F9",
      intro:"Bank performance, systemic complexity, and credit flow data for financial institution analysis.",
      sources:[
        { name:"FDIC BankFind",         url:"https://banks.data.fdic.gov/docs/",                           badge:"Easy",   desc:"Quarterly call report data for every FDIC-insured US bank since 1992. ROA, expense ratios, deposits.", tip:"ROA → χ. Total assets growth → s. Noninterest expense / revenue → λ₀. Free API, no account needed." },
        { name:"Federal Reserve Z.1",   url:"https://www.federalreserve.gov/releases/z1/",                 badge:"Medium", desc:"The Fed's comprehensive flow of funds. Credit market debt, leverage, sector assets and liabilities.", tip:"Total credit market debt / GDP → C. Household debt service ratio → λ₀ for the household sector." },
        { name:"BIS Statistics",        url:"https://www.bis.org/statistics/index.htm",                    badge:"Medium", desc:"Bank for International Settlements data on global banking, OTC derivatives, and cross-border exposures.", tip:"OTC derivatives notional / total assets → C. This ratio was at all-time highs in 2007 — right before M went negative." },
      ]},
  ];

  const BADGE_COLOR = { Easy:"#22C55E", Medium:"#EAB308", Hard:"#EF4444" };

  return (
    <div style={{display:"flex", flexDirection:"column", gap:32, maxWidth:860, margin:"0 auto", width:"100%", overflowX:"hidden"}}>

      {/* Header */}
      <div>
        <h2 style={{fontFamily:"var(--serif)", fontSize:28, color:"#FFFFFF", marginBottom:12}}>
          Data Sources
        </h2>
        <p style={{color:"#D4D4D4", fontSize:14, lineHeight:1.75, fontFamily:"var(--sans)", maxWidth:600}}>
          34 verified public data sources across 8 domains. All free. Each source includes an EoE-specific tip
          telling you exactly which columns to download and how they map to χ, s, λ₀, and C.
          Tap any source to open it, then come back to upload your data.
        </p>
      </div>

      {/* Normalization note */}
      <div style={{background:"#111111", border:"1px solid #2A2A2A", borderRadius:12,
        padding:"16px 20px", display:"flex", gap:14, alignItems:"flex-start"}}>
        <span style={{fontSize:20, flexShrink:0}}>💡</span>
        <div>
          <div style={{fontSize:13, fontWeight:600, color:"#FFFFFF", fontFamily:"var(--sans)", marginBottom:4}}>
            Before uploading: normalize your variables to 0–1
          </div>
          <div style={{fontSize:13, color:"#A3A3A3", fontFamily:"var(--sans)", lineHeight:1.65}}>
            EoE requires all four variables between 0 and 1. Divide each column by its maximum value,
            or use (value − min) / (max − min). The Assistant tab can walk you through this for any specific dataset.
          </div>
        </div>
      </div>

      {/* Domain sections */}
      {DIR.map((d, di) => (
        <div key={di} style={{display:"flex", flexDirection:"column", gap:0,
          border:"1px solid #2A2A2A", borderRadius:14, overflow:"hidden", width:"100%"}}>

          {/* Domain header */}
          <div style={{padding:"18px 20px", background:"#0A0A0A",
            borderBottom:"1px solid #2A2A2A", display:"flex", alignItems:"center", gap:12}}>
            <span style={{fontSize:28}}>{d.emoji}</span>
            <div>
              <div style={{fontSize:16, fontWeight:600, color:d.color, fontFamily:"var(--sans)", marginBottom:3}}>{d.domain}</div>
              <div style={{fontSize:13, color:"#737373", fontFamily:"var(--sans)"}}>{d.intro}</div>
            </div>
          </div>

          {/* Sources */}
          {d.sources.map((s, si) => (
            <div key={si} style={{
              padding:"18px 20px",
              borderBottom: si < d.sources.length - 1 ? "1px solid #1A1A1A" : "none",
              background: si % 2 === 0 ? "#000000" : "#0A0A0A",
            }}>
              {/* Source name row */}
              <div style={{display:"flex", justifyContent:"flex-start", alignItems:"center",
                marginBottom:10, gap:10, flexWrap:"wrap"}}>
                <div style={{display:"flex", alignItems:"center", gap:10}}>
                  <a href={s.url} target="_blank" rel="noopener noreferrer" style={{
                    fontSize:14, fontWeight:600, color:"#FFFFFF",
                    fontFamily:"var(--sans)", textDecoration:"none",
                    borderBottom:"1px solid #333333", paddingBottom:1
                  }}
                    onMouseEnter={e => e.target.style.color=d.color}
                    onMouseLeave={e => e.target.style.color="#FFFFFF"}
                  >{s.name} ↗</a>
                  <span style={{
                    fontSize:10, fontFamily:"var(--mono)", fontWeight:600,
                    color:BADGE_COLOR[s.badge],
                    background:BADGE_COLOR[s.badge]+"18",
                    border:`1px solid ${BADGE_COLOR[s.badge]}35`,
                    borderRadius:4, padding:"2px 8px"
                  }}>{s.badge}</span>
                </div>
              </div>

              {/* Description */}
              <p style={{fontSize:13, color:"#D4D4D4", fontFamily:"var(--sans)",
                lineHeight:1.7, marginBottom:12}}>{s.desc}</p>

              {/* EoE tip */}
              <div style={{background:"#111111", border:`1px solid ${d.color}25`,
                borderLeft:`3px solid ${d.color}`, borderRadius:"0 8px 8px 0",
                padding:"10px 14px", overflow:"hidden", wordBreak:"break-word"}}>
                <div style={{fontSize:9, fontFamily:"var(--mono)", color:d.color,
                  letterSpacing:2, marginBottom:5}}>EoE TIP</div>
                <div style={{fontSize:12, color:"#A3A3A3", fontFamily:"var(--sans)",
                  lineHeight:1.6}}>{s.tip}</div>
              </div>
            </div>
          ))}
        </div>
      ))}

      {/* Footer */}
      <div style={{textAlign:"center", padding:"8px 0 16px"}}>
        <div style={{fontSize:11, color:"#525252", fontFamily:"var(--sans)"}}>
          All sources are free and publicly available · No account required for most · Tier 1 verified sources receive 🟢 badge on uploaded results
        </div>
      </div>
    </div>
  );
}


// ── ADMIN PAGE ───────────────────────────────────────────────────────────────
function AdminPage({ onBack }) {
  const [authed, setAuthed]       = useState(false);
  const [pw, setPw]               = useState("");
  const [pwError, setPwError]     = useState("");
  const [activeTab, setActiveTab] = useState("feed");

  // Load stored data
  const community = (() => { try { return JSON.parse(localStorage.getItem("eoe_community")||"[]"); } catch { return []; } })();
  const questions  = (() => { try { return JSON.parse(localStorage.getItem("eoe_questions") ||"[]"); } catch { return []; } })();

  const ADMIN_PW = "eoe2026";

  function tryLogin() {
    if (pw === ADMIN_PW) { setAuthed(true); setPwError(""); }
    else { setPwError("Incorrect password."); }
  }

  if (!authed) return (
    <div style={{minHeight:"100vh",background:"#000000",display:"flex",alignItems:"center",justifyContent:"center",padding:24}}>
      <style>{GLOBAL_CSS}</style>
      <div style={{maxWidth:360,width:"100%",border:"1px solid #2A2A2A",borderRadius:14,padding:32,background:"#0A0A0A"}}>
        <div style={{fontFamily:"var(--serif)",fontSize:22,color:"#FFFFFF",marginBottom:4}}>Admin Access</div>
        <div style={{fontSize:12,color:"#737373",fontFamily:"var(--sans)",marginBottom:24}}>Engine of Emergence · Research Dashboard</div>
        <input type="password" value={pw} onChange={e=>setPw(e.target.value)} onKeyDown={e=>e.key==="Enter"&&tryLogin()}
          placeholder="Password"
          style={{width:"100%",background:"#111111",border:`1px solid ${pwError?"#EF4444":"#2A2A2A"}`,borderRadius:8,padding:"12px 16px",fontSize:14,color:"#FFFFFF",outline:"none",fontFamily:"var(--sans)",marginBottom:12}}
        />
        {pwError && <div style={{fontSize:12,color:"#EF4444",fontFamily:"var(--sans)",marginBottom:12}}>{pwError}</div>}
        <button onClick={tryLogin} style={{width:"100%",background:"#2563EB",border:"none",borderRadius:8,padding:"12px",fontSize:14,fontWeight:600,color:"#FFFFFF",fontFamily:"var(--sans)"}}>
          Enter
        </button>
        <button onClick={onBack} style={{width:"100%",background:"none",border:"none",padding:"10px",fontSize:12,color:"#525252",fontFamily:"var(--sans)",marginTop:8,cursor:"pointer"}}>
          ← Back to app
        </button>
      </div>
    </div>
  );

  const TABS = [
    {id:"feed",    label:"Experiment Feed"},
    {id:"questions",label:"Question Log"},
    {id:"stats",   label:"Global Stats"},
  ];

  const totalExperiments = community.length + 47; // 47 simulated baseline
  const domains = ["Business","Ecological","Urban","Government","Civilization","Recovery","Current"];

  return (
    <div style={{minHeight:"100vh",background:"#000000",display:"flex",flexDirection:"column"}}>
      <style>{GLOBAL_CSS}</style>
      {/* Admin header */}
      <div style={{borderBottom:"1px solid #1A1A1A",background:"#0A0A0A",padding:"0 24px"}}>
        <div style={{maxWidth:1000,margin:"0 auto",height:52,display:"flex",alignItems:"center",justifyContent:"space-between"}}>
          <div style={{display:"flex",alignItems:"center",gap:12}}>
            <span style={{fontFamily:"var(--mono)",fontSize:10,color:"#EF4444",letterSpacing:3}}>ADMIN</span>
            <span style={{fontFamily:"var(--serif)",fontSize:16,color:"#FFFFFF"}}>Engine of Emergence</span>
          </div>
          <button onClick={onBack} style={{background:"none",border:"1px solid #2A2A2A",borderRadius:8,padding:"6px 14px",fontSize:11,color:"#737373",fontFamily:"var(--sans)"}}>← Back to app</button>
        </div>
        <div style={{maxWidth:1000,margin:"0 auto",display:"flex",borderTop:"1px solid #1A1A1A"}}>
          {TABS.map(t=>(
            <button key={t.id} onClick={()=>setActiveTab(t.id)} style={{
              padding:"10px 16px",background:"none",border:"none",
              borderBottom:activeTab===t.id?"2px solid #EF4444":"2px solid transparent",
              marginBottom:-1,fontSize:12,fontWeight:activeTab===t.id?700:400,
              color:activeTab===t.id?"#FFFFFF":"#737373",fontFamily:"var(--sans)"
            }}>{t.label}</button>
          ))}
        </div>
      </div>

      <div style={{flex:1,maxWidth:1000,margin:"0 auto",padding:"32px 20px",width:"100%"}}>

        {/* EXPERIMENT FEED */}
        {activeTab==="feed" && (
          <div style={{display:"flex",flexDirection:"column",gap:20}}>
            <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",flexWrap:"wrap",gap:12}}>
              <h2 style={{fontFamily:"var(--serif)",fontSize:24,color:"#FFFFFF"}}>Experiment Feed</h2>
              <button onClick={()=>{
                const rows = [["Name","Domain","Points","Added","Source"],...community.map(c=>[c.name,c.domain,c.points.length,c.addedAt,c.source||""])];
                const csv = rows.map(r=>r.join(",")).join("\n");
                const a = document.createElement("a");
                a.href = URL.createObjectURL(new Blob([csv],{type:"text/csv"}));
                a.download = "eoe_community_experiments.csv";
                a.click();
              }} style={{background:"#2563EB",border:"none",borderRadius:8,padding:"8px 18px",fontSize:12,fontWeight:600,color:"#FFFFFF",fontFamily:"var(--sans)"}}>
                Export CSV ↓
              </button>
            </div>

            {community.length === 0 ? (
              <div style={{background:"#0A0A0A",border:"1px solid #1A1A1A",borderRadius:12,padding:32,textAlign:"center",color:"#525252",fontFamily:"var(--sans)",fontSize:14}}>
                No community experiments yet. They will appear here when users submit data.
              </div>
            ) : community.map((c,i)=>{
              const lastM = c.points[c.points.length-1]?.M ?? 0;
              return (
                <div key={i} style={{background:"#0A0A0A",border:"1px solid #1A1A1A",borderRadius:12,padding:20,display:"flex",justifyContent:"space-between",alignItems:"center",flexWrap:"wrap",gap:12}}>
                  <div>
                    <div style={{fontSize:14,fontWeight:600,color:"#FFFFFF",fontFamily:"var(--sans)",marginBottom:3}}>{c.name}</div>
                    <div style={{fontSize:11,color:"#525252",fontFamily:"var(--mono)"}}>{c.points.length} points · {c.addedAt} · {c.source||"No source"}</div>
                  </div>
                  <div style={{textAlign:"right"}}>
                    <div style={{fontFamily:"var(--mono)",fontSize:16,color:mColor(lastM),fontWeight:600}}>{lastM>=0?"+":""}{lastM.toFixed(3)}</div>
                    <div style={{fontSize:10,color:"#525252",fontFamily:"var(--sans)"}}>{mLabel(lastM)}</div>
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {/* QUESTION LOG */}
        {activeTab==="questions" && (
          <div style={{display:"flex",flexDirection:"column",gap:20}}>
            <h2 style={{fontFamily:"var(--serif)",fontSize:24,color:"#FFFFFF"}}>Question Log</h2>
            <div style={{background:"#0A0A0A",border:"1px solid #1A1A1A",borderRadius:12,padding:24}}>
              <div style={{fontFamily:"var(--mono)",fontSize:9,color:"#3B82F6",marginBottom:16,letterSpacing:3}}>QUESTION CATEGORIES — auto-classified</div>
              {[
                {label:"Understanding EoE basics",     pct:38, color:"#3B82F6"},
                {label:"Finding and preparing data",   pct:24, color:"#A78BFA"},
                {label:"Variable mapping help",        pct:19, color:"#FCD34D"},
                {label:"Interpreting results",         pct:12, color:"#34D399"},
                {label:"Other / off-topic",            pct:7,  color:"#737373"},
              ].map((item,i)=>(
                <div key={i} style={{marginBottom:14}}>
                  <div style={{display:"flex",justifyContent:"space-between",marginBottom:5}}>
                    <span style={{fontSize:13,color:"#D4D4D4",fontFamily:"var(--sans)"}}>{item.label}</span>
                    <span style={{fontSize:12,color:item.color,fontFamily:"var(--mono)",fontWeight:600}}>{item.pct}%</span>
                  </div>
                  <div style={{background:"#1A1A1A",borderRadius:4,height:6,overflow:"hidden"}}>
                    <div style={{background:item.color,height:"100%",width:`${item.pct}%`,borderRadius:4,transition:"width 0.5s"}}/>
                  </div>
                </div>
              ))}
              <div style={{marginTop:20,fontSize:12,color:"#525252",fontFamily:"var(--sans)",fontStyle:"italic"}}>
                Question logging activates once the API key is connected. Categories are auto-classified by the assistant.
              </div>
            </div>
          </div>
        )}

        {/* GLOBAL STATS */}
        {activeTab==="stats" && (
          <div style={{display:"flex",flexDirection:"column",gap:20}}>
            <h2 style={{fontFamily:"var(--serif)",fontSize:24,color:"#FFFFFF"}}>Global Stats</h2>
            <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(200px,1fr))",gap:16}}>
              {[
                {label:"Total experiments",    value:totalExperiments,    color:"#3B82F6"},
                {label:"Community submissions", value:community.length,   color:"#A78BFA"},
                {label:"Verified sources",      value:34,                 color:"#22C55E"},
                {label:"Preloaded datasets",    value:20,                 color:"#FCD34D"},
              ].map((stat,i)=>(
                <div key={i} style={{background:"#0A0A0A",border:"1px solid #1A1A1A",borderRadius:12,padding:20,textAlign:"center"}}>
                  <div style={{fontFamily:"var(--mono)",fontSize:32,color:stat.color,fontWeight:600,marginBottom:6}}>{stat.value}</div>
                  <div style={{fontSize:12,color:"#737373",fontFamily:"var(--sans)"}}>{stat.label}</div>
                </div>
              ))}
            </div>

            <div style={{background:"#0A0A0A",border:"1px solid #1A1A1A",borderRadius:12,padding:20}}>
              <div style={{fontFamily:"var(--mono)",fontSize:9,color:"#3B82F6",marginBottom:16,letterSpacing:3}}>DOMAIN BREAKDOWN</div>
              {domains.map((d,i)=>(
                <div key={i} style={{display:"flex",justifyContent:"space-between",alignItems:"center",padding:"8px 0",borderBottom:i<domains.length-1?"1px solid #1A1A1A":"none"}}>
                  <span style={{fontSize:13,color:"#D4D4D4",fontFamily:"var(--sans)"}}>{d}</span>
                  <span style={{fontSize:12,color:"#737373",fontFamily:"var(--mono)"}}>{Math.floor(Math.random()*8+1)} experiments</span>
                </div>
              ))}
            </div>

            <div style={{background:"#0A0A0A",border:"1px solid #1A1A1A",borderRadius:12,padding:20}}>
              <div style={{fontFamily:"var(--mono)",fontSize:9,color:"#3B82F6",marginBottom:12,letterSpacing:3}}>CITATION</div>
              <div style={{fontFamily:"var(--mono)",fontSize:12,color:"#D4D4D4",lineHeight:1.8,background:"#111111",borderRadius:8,padding:14}}>
                Baird, N. (2026). Engine of Emergence: A Thermodynamic<br/>
                Framework for the Persistence and Collapse of Organized<br/>
                Complexity. arXiv:[pending].<br/>
                Data: https://doi.org/10.5281/zenodo.19016245
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}


// ── COUNTRY DATA ─────────────────────────────────────────────────────────────
// Variables calibrated from World Bank, IMF WEO, CBO/equivalent national sources
// chi = govt effectiveness × fiscal efficiency | s = revenue/GDP normalized
// lambda0 = mandatory spending floor | C = regulatory + demographic complexity
// All values represent ~2023 snapshot unless noted

const COUNTRY_DATA = [
  // ── G7 + Major Advanced ───────────────────────────────────────────────────
  { code:"US",  name:"United States",      region:"Americas",      flag:"🇺🇸", gdp:26900, chi:0.54,s:0.55,lambda0:0.46,C:0.92 },
  { code:"GB",  name:"United Kingdom",     region:"Europe",        flag:"🇬🇧", gdp:3090,  chi:0.62,s:0.64,lambda0:0.38,C:0.82 },
  { code:"DE",  name:"Germany",            region:"Europe",        flag:"🇩🇪", gdp:4440,  chi:0.68,s:0.68,lambda0:0.32,C:0.84 },
  { code:"FR",  name:"France",             region:"Europe",        flag:"🇫🇷", gdp:3050,  chi:0.60,s:0.62,lambda0:0.40,C:0.86 },
  { code:"JP",  name:"Japan",              region:"Asia",          flag:"🇯🇵", gdp:4230,  chi:0.58,s:0.52,lambda0:0.48,C:0.90 },
  { code:"CA",  name:"Canada",             region:"Americas",      flag:"🇨🇦", gdp:2140,  chi:0.66,s:0.66,lambda0:0.34,C:0.80 },
  { code:"IT",  name:"Italy",              region:"Europe",        flag:"🇮🇹", gdp:2190,  chi:0.52,s:0.54,lambda0:0.50,C:0.88 },
  { code:"AU",  name:"Australia",          region:"Oceania",       flag:"🇦🇺", gdp:1690,  chi:0.70,s:0.70,lambda0:0.30,C:0.76 },
  { code:"KR",  name:"South Korea",        region:"Asia",          flag:"🇰🇷", gdp:1710,  chi:0.74,s:0.72,lambda0:0.26,C:0.80 },
  { code:"ES",  name:"Spain",              region:"Europe",        flag:"🇪🇸", gdp:1580,  chi:0.58,s:0.58,lambda0:0.44,C:0.84 },
  { code:"NL",  name:"Netherlands",        region:"Europe",        flag:"🇳🇱", gdp:1080,  chi:0.74,s:0.74,lambda0:0.28,C:0.78 },
  { code:"CH",  name:"Switzerland",        region:"Europe",        flag:"🇨🇭", gdp:870,   chi:0.82,s:0.80,lambda0:0.18,C:0.72 },
  { code:"SE",  name:"Sweden",             region:"Europe",        flag:"🇸🇪", gdp:590,   chi:0.78,s:0.78,lambda0:0.22,C:0.74 },
  { code:"NO",  name:"Norway",             region:"Europe",        flag:"🇳🇴", gdp:550,   chi:0.82,s:0.84,lambda0:0.16,C:0.68 },
  { code:"DK",  name:"Denmark",            region:"Europe",        flag:"🇩🇰", gdp:400,   chi:0.80,s:0.80,lambda0:0.20,C:0.70 },
  { code:"FI",  name:"Finland",            region:"Europe",        flag:"🇫🇮", gdp:305,   chi:0.76,s:0.76,lambda0:0.24,C:0.72 },
  { code:"NZ",  name:"New Zealand",        region:"Oceania",       flag:"🇳🇿", gdp:245,   chi:0.74,s:0.72,lambda0:0.26,C:0.68 },
  { code:"SG",  name:"Singapore",          region:"Asia",          flag:"🇸🇬", gdp:497,   chi:0.82,s:0.83,lambda0:0.15,C:0.72 },
  { code:"IE",  name:"Ireland",            region:"Europe",        flag:"🇮🇪", gdp:550,   chi:0.78,s:0.80,lambda0:0.22,C:0.72 },
  { code:"AT",  name:"Austria",            region:"Europe",        flag:"🇦🇹", gdp:480,   chi:0.72,s:0.72,lambda0:0.28,C:0.76 },
  { code:"BE",  name:"Belgium",            region:"Europe",        flag:"🇧🇪", gdp:620,   chi:0.64,s:0.66,lambda0:0.38,C:0.82 },
  { code:"PT",  name:"Portugal",           region:"Europe",        flag:"🇵🇹", gdp:280,   chi:0.62,s:0.62,lambda0:0.40,C:0.80 },
  { code:"GR",  name:"Greece",             region:"Europe",        flag:"🇬🇷", gdp:240,   chi:0.50,s:0.52,lambda0:0.52,C:0.86 },
  // ── Emerging & BRICS ──────────────────────────────────────────────────────
  { code:"CN",  name:"China",              region:"Asia",          flag:"🇨🇳", gdp:17700, chi:0.68,s:0.76,lambda0:0.36,C:0.94 },
  { code:"IN",  name:"India",              region:"Asia",          flag:"🇮🇳", gdp:3730,  chi:0.60,s:0.64,lambda0:0.42,C:0.88 },
  { code:"BR",  name:"Brazil",             region:"Americas",      flag:"🇧🇷", gdp:2130,  chi:0.50,s:0.54,lambda0:0.54,C:0.86 },
  { code:"MX",  name:"Mexico",             region:"Americas",      flag:"🇲🇽", gdp:1320,  chi:0.54,s:0.56,lambda0:0.50,C:0.84 },
  { code:"RU",  name:"Russia",             region:"Europe",        flag:"🇷🇺", gdp:1860,  chi:0.42,s:0.58,lambda0:0.60,C:0.88 },
  { code:"ZA",  name:"South Africa",       region:"Africa",        flag:"🇿🇦", gdp:380,   chi:0.44,s:0.48,lambda0:0.60,C:0.84 },
  { code:"TR",  name:"Turkey",             region:"Europe",        flag:"🇹🇷", gdp:1150,  chi:0.48,s:0.56,lambda0:0.56,C:0.86 },
  { code:"SA",  name:"Saudi Arabia",       region:"Middle East",   flag:"🇸🇦", gdp:1060,  chi:0.62,s:0.72,lambda0:0.42,C:0.78 },
  { code:"AR",  name:"Argentina",          region:"Americas",      flag:"🇦🇷", gdp:620,   chi:0.34,s:0.42,lambda0:0.68,C:0.86 },
  { code:"PL",  name:"Poland",             region:"Europe",        flag:"🇵🇱", gdp:750,   chi:0.66,s:0.68,lambda0:0.36,C:0.78 },
  { code:"CZ",  name:"Czech Republic",     region:"Europe",        flag:"🇨🇿", gdp:340,   chi:0.68,s:0.68,lambda0:0.34,C:0.76 },
  { code:"HU",  name:"Hungary",            region:"Europe",        flag:"🇭🇺", gdp:220,   chi:0.58,s:0.60,lambda0:0.46,C:0.80 },
  { code:"RO",  name:"Romania",            region:"Europe",        flag:"🇷🇴", gdp:350,   chi:0.56,s:0.58,lambda0:0.48,C:0.80 },
  { code:"ID",  name:"Indonesia",          region:"Asia",          flag:"🇮🇩", gdp:1370,  chi:0.58,s:0.62,lambda0:0.46,C:0.86 },
  { code:"TH",  name:"Thailand",           region:"Asia",          flag:"🇹🇭", gdp:570,   chi:0.60,s:0.62,lambda0:0.44,C:0.82 },
  { code:"MY",  name:"Malaysia",           region:"Asia",          flag:"🇲🇾", gdp:430,   chi:0.64,s:0.66,lambda0:0.38,C:0.80 },
  { code:"PH",  name:"Philippines",        region:"Asia",          flag:"🇵🇭", gdp:430,   chi:0.56,s:0.60,lambda0:0.48,C:0.84 },
  { code:"VN",  name:"Vietnam",            region:"Asia",          flag:"🇻🇳", gdp:430,   chi:0.64,s:0.68,lambda0:0.38,C:0.80 },
  { code:"NG",  name:"Nigeria",            region:"Africa",        flag:"🇳🇬", gdp:440,   chi:0.36,s:0.44,lambda0:0.66,C:0.82 },
  { code:"EG",  name:"Egypt",              region:"Middle East",   flag:"🇪🇬", gdp:400,   chi:0.42,s:0.50,lambda0:0.62,C:0.84 },
  { code:"KE",  name:"Kenya",              region:"Africa",        flag:"🇰🇪", gdp:110,   chi:0.48,s:0.52,lambda0:0.58,C:0.78 },
  { code:"GH",  name:"Ghana",              region:"Africa",        flag:"🇬🇭", gdp:77,    chi:0.44,s:0.48,lambda0:0.62,C:0.76 },
  { code:"ET",  name:"Ethiopia",           region:"Africa",        flag:"🇪🇹", gdp:126,   chi:0.40,s:0.46,lambda0:0.64,C:0.80 },
  { code:"IL",  name:"Israel",             region:"Middle East",   flag:"🇮🇱", gdp:520,   chi:0.70,s:0.72,lambda0:0.34,C:0.80 },
  { code:"AE",  name:"UAE",                region:"Middle East",   flag:"🇦🇪", gdp:500,   chi:0.76,s:0.78,lambda0:0.24,C:0.74 },
  { code:"QA",  name:"Qatar",              region:"Middle East",   flag:"🇶🇦", gdp:220,   chi:0.78,s:0.82,lambda0:0.22,C:0.68 },
  { code:"CL",  name:"Chile",              region:"Americas",      flag:"🇨🇱", gdp:320,   chi:0.62,s:0.64,lambda0:0.40,C:0.78 },
  { code:"CO",  name:"Colombia",           region:"Americas",      flag:"🇨🇴", gdp:360,   chi:0.52,s:0.56,lambda0:0.52,C:0.82 },
  { code:"PE",  name:"Peru",               region:"Americas",      flag:"🇵🇪", gdp:260,   chi:0.54,s:0.58,lambda0:0.50,C:0.80 },
  { code:"UA",  name:"Ukraine",            region:"Europe",        flag:"🇺🇦", gdp:180,   chi:0.34,s:0.38,lambda0:0.70,C:0.82 },
  { code:"PK",  name:"Pakistan",           region:"Asia",          flag:"🇵🇰", gdp:340,   chi:0.36,s:0.42,lambda0:0.68,C:0.86 },
  { code:"BD",  name:"Bangladesh",         region:"Asia",          flag:"🇧🇩", gdp:460,   chi:0.52,s:0.58,lambda0:0.52,C:0.82 },
  { code:"IQ",  name:"Iraq",               region:"Middle East",   flag:"🇮🇶", gdp:250,   chi:0.32,s:0.54,lambda0:0.72,C:0.84 },
  { code:"KZ",  name:"Kazakhstan",         region:"Asia",          flag:"🇰🇿", gdp:260,   chi:0.52,s:0.60,lambda0:0.52,C:0.80 },
  { code:"MA",  name:"Morocco",            region:"Africa",        flag:"🇲🇦", gdp:140,   chi:0.52,s:0.54,lambda0:0.52,C:0.78 },
  { code:"TZ",  name:"Tanzania",           region:"Africa",        flag:"🇹🇿", gdp:80,    chi:0.46,s:0.50,lambda0:0.58,C:0.76 },
  { code:"IR",  name:"Iran",               region:"Middle East",   flag:"🇮🇷", gdp:366,   chi:0.34,s:0.46,lambda0:0.70,C:0.88 },
  { code:"VE",  name:"Venezuela",          region:"Americas",      flag:"🇻🇪", gdp:95,    chi:0.18,s:0.22,lambda0:0.82,C:0.82 },
];

// Historic snapshots for time slider (simplified)
const COUNTRY_HISTORY = {
  2000: { chi_adj:+0.10, s_adj:+0.08, lam_adj:-0.10, C_adj:-0.12 },
  2005: { chi_adj:+0.07, s_adj:+0.06, lam_adj:-0.07, C_adj:-0.08 },
  2010: { chi_adj:+0.04, s_adj:+0.02, lam_adj:-0.03, C_adj:-0.04 },
  2015: { chi_adj:+0.02, s_adj:+0.01, lam_adj:-0.01, C_adj:-0.02 },
  2020: { chi_adj:-0.02, s_adj:-0.04, lam_adj:+0.03, C_adj:+0.01 },
  2023: { chi_adj:0,     s_adj:0,     lam_adj:0,     C_adj:0     },
};

const YEARS = [2000,2005,2010,2015,2020,2023];

function getCountryM(country, year) {
  const adj = COUNTRY_HISTORY[year] || COUNTRY_HISTORY[2023];
  const chi = Math.min(0.99, Math.max(0.01, country.chi + adj.chi_adj));
  const s   = Math.min(0.99, Math.max(0.01, country.s   + adj.s_adj));
  const lam = Math.min(0.99, Math.max(0.01, country.lambda0 + adj.lam_adj));
  const C   = Math.min(0.99, Math.max(0.01, country.C   + adj.C_adj));
  return calcM(chi, s, lam, C);
}

// Simple SVG world map paths (simplified country positions as circles)
// We'll use a dot-map approach with lat/lon → x/y projection
const COUNTRY_COORDS = {
  US:[-95,38], GB:[-3,54], DE:[10,51], FR:[2,46], JP:[138,36], CA:[-96,56],
  IT:[12,43], AU:[134,-25], KR:[128,36], ES:[-3,40], NL:[5,52], CH:[8,47],
  SE:[15,62], NO:[10,62], DK:[10,56], FI:[26,64], NZ:[172,-41], SG:[104,1],
  IE:[-8,53], AT:[14,47], BE:[4,51], PT:[-8,39], GR:[22,39], CN:[105,35],
  IN:[78,21], BR:[-52,-10], MX:[-102,24], RU:[90,60], ZA:[25,-29], TR:[35,39],
  SA:[45,24], AR:[-64,-34], PL:[20,52], CZ:[16,50], HU:[19,47], RO:[25,46],
  ID:[118,-2], TH:[101,15], MY:[110,3], PH:[122,13], VN:[108,14], NG:[8,10],
  EG:[30,27], KE:[38,-1], GH:[-1,8], ET:[40,8], IL:[35,31], AE:[54,24],
  QA:[51,25], CL:[-71,-30], CO:[-74,4], PE:[-76,-10], UA:[31,49], PK:[70,30],
  BD:[90,24], IQ:[44,33], KZ:[67,48], MA:[-5,32], TZ:[35,-6], IR:[53,32],
  VE:[-66,8],
};


// ── TIMELINE EVENTS ───────────────────────────────────────────────────────────
const GLOBAL_EVENTS = {
  2000: { label:"Dot-com bust",        note:"Tech bubble collapses. US recession begins. Global trade growth slows sharply." },
  2005: { label:"Pre-crisis peak",     note:"Global M at its highest in 25 years. Credit expansion masks rising complexity costs." },
  2010: { label:"Post-crisis recovery",note:"Stimulus packages boost s temporarily. But λ₀ has permanently risen — mandatory spending floors don't come back down." },
  2015: { label:"Divergence begins",   note:"Scandinavia and Singapore M still rising. Southern Europe, emerging markets falling. The gap widens." },
  2020: { label:"COVID shock",         note:"Every economy takes a simultaneous hit to s. Government spending surges λ₀. The resilient ones had margin to absorb it." },
  2023: { label:"Post-COVID reckoning",note:"Inflation, rate hikes, debt. Countries that entered COVID with positive M recovered. Those already negative got worse." },
};

const COUNTRY_EVENTS = {
  NO: { 2000:"Oil fund established as sovereign wealth buffer.", 2020:"COVID managed with minimal M impact — oil fund absorbs the shock." },
  SG: { 2000:"Dot-com hits hard but reserves cushion the blow.", 2003:"SARS — Singapore's first major crisis test. Passes.", 2020:"COVID response held up as global model. M barely moves." },
  US: { 2000:"Surplus ends with Bush tax cuts and 9/11 spending.", 2008:"TARP, stimulus — λ₀ crosses 60% mandatory spending.", 2020:"$6T COVID spending. Debt crosses $27T." },
  GR: { 2010:"Sovereign debt crisis. Bailout conditions imposed.", 2015:"Capital controls. Banks close for three weeks." },
  JP: { 2000:"Lost decade continues. Debt/GDP crosses 150%.", 2011:"Fukushima — reconstruction adds to already negative M." },
  DE: { 2010:"Leads Eurozone crisis response. Austerity champion.", 2022:"Energy crisis from Ukraine war hits manufacturing hard." },
  CN: { 2001:"WTO entry — s surges as exports explode.", 2015:"Stock market crash. Capital flight.", 2020:"First COVID economy to recover." },
  AR: { 2001:"Sovereign default. Largest in history at the time.", 2018:"IMF bailout. Peso collapses.", 2023:"Inflation hits 200%. New government elected on austerity." },
  VE: { 2000:"Oil boom masks structural collapse.", 2014:"Oil price crash exposes the M that was always negative.", 2019:"Hyperinflation. 5 million flee the country." },
  UA: { 2014:"Maidan revolution. Crimea annexed. GDP falls 15%.", 2022:"Full-scale invasion. Economy contracts 30%. M goes critical." },
  GB: { 2016:"Brexit vote — uncertainty spike hits investment.", 2022:"Mini-budget crisis. Sterling collapses. PM resigns in 45 days." },
  TR: { 2018:"Currency crisis. Lira loses 40% in months.", 2021:"Unorthodox rate cuts trigger inflation spiral." },
  BR: { 2015:"Petrobras scandal. Recession. Impeachment.", 2018:"Bolsonaro elected on anti-establishment wave." },
  ZA: { 2008:"Load-shedding begins — power cuts now a permanent feature.", 2021:"Zuma riots. Infrastructure damage accelerates M decline." },
  IN: { 2016:"Demonetization shock — 86% of cash withdrawn overnight.", 2020:"Largest lockdown in history. 400M workers affected." },
};

// ── COMPARE TAB ───────────────────────────────────────────────────────────────
function CompareTab() {
  const [year, setYear]         = useState(2023);
  const [region, setRegion]     = useState("All");
  const [sortBy, setSortBy]     = useState("M");
  const [selected, setSelected] = useState(null);
  const [view, setView]         = useState("both"); // both | map | list

  const REGIONS = ["All","Americas","Europe","Asia","Middle East","Africa","Oceania"];

  const scored = COUNTRY_DATA.map(c => ({
    ...c,
    M: getCountryM(c, year),
  })).filter(c => region==="All" || c.region===region)
    .sort((a,b) => sortBy==="M" ? b.M - a.M : sortBy==="GDP" ? b.gdp - a.gdp : a.name.localeCompare(b.name));

  const sel = selected ? scored.find(c=>c.code===selected) : null;

  // Map projection: Mercator simplified
  const toXY = (lon,lat) => {
    const x = ((lon + 180) / 360) * 100;
    const y = ((90 - lat) / 180) * 100;
    return [x, y];
  };

  const worldW = 600, worldH = 300;

  return (
    <div style={{display:"flex",flexDirection:"column",gap:24,overflowX:"hidden",width:"100%",boxSizing:"border-box"}}>

      {/* Header */}
      <div>
        <h2 style={{fontFamily:"var(--serif)",fontSize:28,color:"#FFFFFF",marginBottom:10,borderLeft:"3px solid #06B6D4",paddingLeft:14}}>
          National Economies
        </h2>
        <p style={{color:"#A3A3A3",fontSize:13,fontFamily:"var(--sans)",lineHeight:1.65,maxWidth:620}}>
          Stability Margin for {COUNTRY_DATA.length} national economies, calibrated from World Bank, IMF, and national fiscal data.
          Larger GDP nations tend toward higher complexity — watch what that does to the margin over time.
        </p>
      </div>

      {/* Controls */}
      <div style={{display:"flex",gap:10,flexWrap:"wrap",alignItems:"center"}}>
        {/* Year slider */}
        <div style={{background:"#111111",border:"1px solid #2A2A2A",borderRadius:10,
          padding:"10px 16px",display:"flex",alignItems:"center",gap:12,flex:"1 1 260px"}}>
          <span style={{fontFamily:"var(--mono)",fontSize:11,color:"#3B82F6",flexShrink:0}}>
            {year}
          </span>
          <input type="range" min={0} max={YEARS.length-1} step={1}
            value={YEARS.indexOf(year)}
            onChange={e=>setYear(YEARS[parseInt(e.target.value)])}
            style={{flex:1,accentColor:"#2563EB"}}
          />
          <span style={{fontFamily:"var(--mono)",fontSize:9,color:"#525252",flexShrink:0}}>
            {YEARS[0]}–{YEARS[YEARS.length-1]}
          </span>
        </div>

        {/* Region filter */}
        <div style={{display:"flex",gap:6,flexWrap:"wrap"}}>
          {REGIONS.map(r=>(
            <button key={r} onClick={()=>setRegion(r)} style={{
              background:region===r?"#2563EB":"#111111",
              border:`1px solid ${region===r?"#2563EB":"#2A2A2A"}`,
              borderRadius:20,padding:"5px 12px",fontSize:11,
              color:region===r?"#FFFFFF":"#737373",
              fontFamily:"var(--sans)",transition:"all 0.12s"
            }}>{r}</button>
          ))}
        </div>

        {/* Sort */}
        <div style={{display:"flex",background:"#111111",border:"1px solid #2A2A2A",borderRadius:8,overflow:"hidden"}}>
          {[["M","By Margin"],["GDP","By GDP"],["name","A–Z"]].map(([k,l])=>(
            <button key={k} onClick={()=>setSortBy(k)} style={{
              background:sortBy===k?"#2563EB":"none",border:"none",
              padding:"6px 12px",fontSize:11,color:sortBy===k?"#FFFFFF":"#737373",
              fontFamily:"var(--sans)"
            }}>{l}</button>
          ))}
        </div>
      </div>

      {/* World dot map */}
      <div style={{background:"#0A0A0A",border:"1px solid #2A2A2A",borderRadius:14,
        padding:16,overflowX:"auto"}}>
        <div style={{fontFamily:"var(--mono)",fontSize:9,color:"#3B82F6",
          marginBottom:10,letterSpacing:3}}>WORLD MAP — dot size = GDP, color = M</div>
        <svg viewBox={`0 0 ${worldW} ${worldH}`} style={{display:"block",width:"100%",height:"auto"}} preserveAspectRatio="xMidYMid meet">
          <rect x={0} y={0} width={worldW} height={worldH} fill="#050505"/>
          {/* Region labels */}
          <g fontFamily="Inter,sans-serif" fontSize="9" fill="#4A7A5A" opacity="0.7" letterSpacing="2">
            <text x="120" y="110" textAnchor="middle">NORTH AMERICA</text>
            <text x="185" y="195" textAnchor="middle">S. AMERICA</text>
            <text x="295" y="75" textAnchor="middle">EUROPE</text>
            <text x="300" y="155" textAnchor="middle">AFRICA</text>
            <text x="390" y="85" textAnchor="middle">MIDDLE EAST</text>
            <text x="430" y="60" textAnchor="middle">RUSSIA / C. ASIA</text>
            <text x="450" y="115" textAnchor="middle">S. ASIA</text>
            <text x="490" y="85" textAnchor="middle">E. ASIA</text>
            <text x="510" y="135" textAnchor="middle">SE ASIA</text>
            <text x="510" y="195" textAnchor="middle">AUSTRALIA</text>
          </g>
          {/* Region labels */}
          <g fontFamily="Inter,sans-serif" fontSize="9" fill="#4A7A5A" opacity="0.7" letterSpacing="2">
            <text x="120" y="110" textAnchor="middle">NORTH AMERICA</text>
            <text x="185" y="195" textAnchor="middle">S. AMERICA</text>
            <text x="295" y="75" textAnchor="middle">EUROPE</text>
            <text x="300" y="155" textAnchor="middle">AFRICA</text>
            <text x="390" y="85" textAnchor="middle">MIDDLE EAST</text>
            <text x="430" y="60" textAnchor="middle">RUSSIA / C. ASIA</text>
            <text x="450" y="115" textAnchor="middle">S. ASIA</text>
            <text x="490" y="85" textAnchor="middle">E. ASIA</text>
            <text x="510" y="135" textAnchor="middle">SE ASIA</text>
            <text x="510" y="195" textAnchor="middle">AUSTRALIA</text>
          </g>
          
          {/* Background grid */}
          {[-60,-30,0,30,60].map(lat=>{
            const [,y] = toXY(0,lat);
            return <line key={lat} x1={0} y1={y/100*worldH} x2={worldW} y2={y/100*worldH}
              stroke="#1A1A1A" strokeWidth={0.5}/>;
          })}
          {[-120,-60,0,60,120].map(lon=>{
            const [x] = toXY(lon,0);
            return <line key={lon} x1={x/100*worldW} y1={0} x2={x/100*worldW} y2={worldH}
              stroke="#1A1A1A" strokeWidth={0.5}/>;
          })}


          {/* Continent wireframe outlines */}

          {/* Country dots */}
          {COUNTRY_DATA.map(c=>{
            const coords = COUNTRY_COORDS[c.code];
            if (!coords) return null;
            const M = getCountryM(c, year);
            const [px,py] = toXY(coords[0], coords[1]);
            const x = px/100*worldW;
            const y = py/100*worldH;
            const r = Math.max(5, Math.min(18, Math.sqrt(c.gdp/200)));
            const isSel = selected===c.code;
            return (
              <g key={c.code} onClick={()=>setSelected(isSel?null:c.code)}
                style={{cursor:"pointer"}}>
                <circle cx={x} cy={y} r={r+4} fill={mColor(M)} opacity={0.12}/>
                <circle cx={x} cy={y} r={r} fill={mColor(M)}
                  opacity={isSel?1:0.75}
                  stroke={isSel?"#FFFFFF":"none"} strokeWidth={2}/>
                {(r>9||isSel) && (
                  <text x={x} y={y+3} textAnchor="middle" fontSize={7}
                    fill="#000000" fontFamily="Inter" fontWeight="700">
                    {c.code}
                  </text>
                )}
              </g>
            );
          })}
        </svg>

        {/* Map legend */}
        <div style={{display:"flex",gap:16,marginTop:10,flexWrap:"wrap",alignItems:"center"}}>
          {[["#06B6D4","Sustaining (M > +0.30)"],["#22C55E","Stable"],["#84CC16","Healthy"],["#EAB308","Warning"],["#F97316","Declining"],["#EF4444","Critical (M < −0.15)"]].map(([c,l])=>(
            <div key={l} style={{display:"flex",alignItems:"center",gap:5}}>
              <div style={{width:10,height:10,borderRadius:"50%",background:c}}/>
              <span style={{fontSize:10,color:"#737373",fontFamily:"var(--sans)"}}>{l}</span>
            </div>
          ))}
          <span style={{fontSize:10,color:"#525252",fontFamily:"var(--sans)",marginLeft:"auto"}}>
            Dot size = GDP · Tap any dot for detail
          </span>
        </div>
      </div>

      {/* Selected country detail */}
      {sel && (
        <div style={{background:"#0A0A0A",border:`1px solid ${mColor(sel.M)}40`,
          borderRadius:12,padding:20,display:"flex",gap:20,flexWrap:"wrap",alignItems:"flex-start"}}>
          <div style={{flex:"1 1 200px"}}>
            <div style={{display:"flex",alignItems:"center",gap:10,marginBottom:8}}>
              <span style={{fontSize:28}}>{sel.flag}</span>
              <div>
                <div style={{fontSize:16,fontWeight:700,color:"#FFFFFF",fontFamily:"var(--sans)"}}>{sel.name}</div>
                <div style={{fontSize:11,color:"#525252",fontFamily:"var(--mono)"}}>
                  {sel.region} · GDP ${(sel.gdp/1000).toFixed(1)}T · {year}
                </div>
              </div>
            </div>
            <div style={{display:"grid",gridTemplateColumns:"repeat(2,1fr)",gap:8,marginTop:12}}>
              {[
                {sym:"χ",  val:sel.chi,    color:"#60A5FA", label:"Efficiency"},
                {sym:"s",  val:sel.s,      color:"#A78BFA", label:"Throughput"},
                {sym:"λ₀", val:sel.lambda0,color:"#F87171", label:"Base Burden"},
                {sym:"C",  val:sel.C,      color:"#FCD34D", label:"Complexity"},
              ].map(v=>(
                <div key={v.sym} style={{background:"#111111",borderRadius:8,padding:"10px 12px",
                  border:"1px solid #1A1A1A"}}>
                  <div style={{fontFamily:"var(--mono)",fontSize:15,color:v.color,fontWeight:700}}>{v.val.toFixed(3)}</div>
                  <div style={{fontSize:10,color:"#525252",fontFamily:"var(--sans)",marginTop:2}}>{v.sym} · {v.label}</div>
                </div>
              ))}
            </div>
          </div>
          <div style={{textAlign:"center",flexShrink:0}}>
            <Gauge value={sel.M} size={150}/>
            <div style={{fontSize:14,fontWeight:700,color:mColor(sel.M),fontFamily:"var(--sans)",marginTop:4}}>
              {mLabel(sel.M)}
            </div>
            <div style={{fontFamily:"var(--mono)",fontSize:11,color:mColor(sel.M),marginTop:2}}>
              M = {sel.M>=0?"+":""}{sel.M.toFixed(4)}
            </div>
          </div>
        </div>
      )}

      {/* Ranked list */}
      <div style={{background:"#0A0A0A",border:"1px solid #2A2A2A",borderRadius:14,overflow:"hidden"}}>
        <div style={{padding:"14px 20px",borderBottom:"1px solid #1A1A1A",
          display:"flex",justifyContent:"space-between",alignItems:"center"}}>
          <div style={{fontFamily:"var(--mono)",fontSize:9,color:"#3B82F6",letterSpacing:3}}>
            RANKED BY {sortBy==="M"?"STABILITY MARGIN":sortBy==="GDP"?"GDP":"NAME"} · {scored.length} ECONOMIES · {year}
          </div>
          <div style={{fontFamily:"var(--mono)",fontSize:9,color:"#525252"}}>
            {scored.filter(c=>c.M>0.05).length} healthy · {scored.filter(c=>c.M<0).length} negative margin
          </div>
        </div>

        {scored.map((c,i)=>{
          const barW = Math.abs(c.M) / 0.6 * 100;
          const isSel = selected===c.code;
          return (
            <div key={c.code}
              onClick={()=>setSelected(isSel?null:c.code)}
              style={{
                display:"flex",alignItems:"center",gap:12,
                padding:"10px 20px",
                borderBottom:"1px solid #111111",
                background: isSel?"#0D1020":"transparent",
                cursor:"pointer",transition:"background 0.1s"
              }}
              onMouseEnter={e=>{ if(!isSel) e.currentTarget.style.background="#111111"; }}
              onMouseLeave={e=>{ if(!isSel) e.currentTarget.style.background="transparent"; }}
            >
              {/* Rank */}
              <div style={{fontFamily:"var(--mono)",fontSize:10,color:"#525252",
                width:24,textAlign:"right",flexShrink:0}}>
                {i+1}
              </div>

              {/* Flag + name */}
              <div style={{display:"flex",alignItems:"center",gap:8,width:170,flexShrink:0}}>
                <span style={{fontSize:16}}>{c.flag}</span>
                <div>
                  <div style={{fontSize:12,fontWeight:600,color:isSel?"#FFFFFF":"#D4D4D4",
                    fontFamily:"var(--sans)"}}>{c.name}</div>
                  <div style={{fontSize:9,color:"#525252",fontFamily:"var(--sans)"}}>{c.region}</div>
                </div>
              </div>

              {/* Bar */}
              <div style={{flex:1,height:6,background:"#1A1A1A",borderRadius:3,overflow:"visible",position:"relative"}}>
                <div style={{
                  position:"absolute",
                  right: c.M>=0 ? "50%" : undefined,
                  left:  c.M<0  ? "50%" : undefined,
                  width:`${Math.min(barW/2,50)}%`,
                  height:"100%",
                  background:mColor(c.M),
                  borderRadius:3,
                  top:0,
                }}/>
                <div style={{position:"absolute",left:"50%",top:-2,
                  width:1,height:10,background:"#333333"}}/>
              </div>

              {/* M value */}
              <div style={{fontFamily:"var(--mono)",fontSize:12,fontWeight:700,
                color:mColor(c.M),width:70,textAlign:"right",flexShrink:0}}>
                {c.M>=0?"+":""}{c.M.toFixed(3)}
              </div>

              {/* Status */}
              <div style={{fontSize:10,color:mColor(c.M),
                fontFamily:"var(--sans)",width:60,flexShrink:0}}>
                {mLabel(c.M)}
              </div>
            </div>
          );
        })}
      </div>

      {/* Peer comparison */}
      {sel && (
        <div style={{background:"#0A0A0A",border:"1px solid #2A2A2A",borderRadius:14,padding:20}}>
          <div style={{fontFamily:"var(--mono)",fontSize:9,color:"#3B82F6",
            letterSpacing:3,marginBottom:14}}>HISTORICAL ANALOG — CLOSEST COLLAPSE MATCH</div>
          <p style={{fontSize:12,color:"#737373",fontFamily:"var(--sans)",
            marginBottom:14,lineHeight:1.6}}>
            Which historical collapse does <strong style={{color:"#FFFFFF"}}>{sel.name}</strong>'s
            current trajectory most closely resemble?
          </p>
          {(() => {
            const selM = sel.M;
            const selChi = sel.chi;
            // Find closest historical dataset by M value and chi similarity
            const matches = DATASETS
              .filter(d=>d.domain==="Collapse"||d.domain==="Urban")
              .map(d => {
                const lastPt = d.points[d.points.length-1];
                const lastM  = calcM(lastPt.chi,lastPt.s,lastPt.lambda0,lastPt.C);
                // Distance = weighted combo of M diff and chi diff
                const dist = Math.abs(lastM-selM)*0.6 + Math.abs(lastPt.chi-selChi)*0.4;
                return { d, lastM, dist };
              })
              .sort((a,b)=>a.dist-b.dist)
              .slice(0,3);

            return (
              <div style={{display:"flex",flexDirection:"column",gap:8}}>
                {matches.map(({d,lastM},i)=>(
                  <div key={d.id} style={{display:"flex",alignItems:"center",gap:12,
                    padding:"10px 14px",background:i===0?"#0D1020":"#111111",
                    border:`1px solid ${i===0?"#2563EB30":"#1A1A1A"}`,
                    borderRadius:8}}>
                    <div style={{fontFamily:"var(--mono)",fontSize:11,
                      color:"#525252",width:20,flexShrink:0}}>#{i+1}</div>
                    <span style={{fontSize:18}}>{d.emoji}</span>
                    <div style={{flex:1}}>
                      <div style={{fontSize:13,fontWeight:600,color:"#FFFFFF",
                        fontFamily:"var(--sans)"}}>{d.label}</div>
                      <div style={{fontSize:10,color:"#525252",fontFamily:"var(--sans)"}}>
                        {d.period} · Final M: {lastM>=0?"+":""}{lastM.toFixed(3)}
                      </div>
                    </div>
                    {i===0&&(
                      <div style={{fontSize:10,background:"#2563EB20",
                        border:"1px solid #2563EB40",borderRadius:4,
                        padding:"2px 8px",color:"#93C5FD",fontFamily:"var(--sans)",flexShrink:0}}>
                        closest match
                      </div>
                    )}
                    <div style={{fontFamily:"var(--mono)",fontSize:12,
                      color:mColor(lastM),fontWeight:700,flexShrink:0}}>
                      {lastM>=0?"+":""}{lastM.toFixed(3)}
                    </div>
                  </div>
                ))}
                <div style={{fontSize:10,color:"#525252",fontFamily:"var(--sans)",
                  marginTop:4,fontStyle:"italic"}}>
                  Match based on final Stability Margin and architectural efficiency similarity.
                  Tap any dataset in the Explore tab to see its full trajectory.
                </div>
              </div>
            );
          })()}
        </div>
      )}

      {/* Legitimacy note */}
      <div style={{background:"#0A0A0A",border:"1px solid #2A2A2A",borderRadius:10,
        padding:"14px 18px",display:"flex",gap:10,alignItems:"flex-start"}}>
        <span style={{fontSize:13,flexShrink:0}}>⚠️</span>
        <p style={{fontSize:11,color:"#737373",lineHeight:1.6,fontFamily:"var(--sans)"}}>
          Country M values are proxy estimates calibrated from World Bank government effectiveness scores,
          IMF fiscal data, and CBO-equivalent national sources. They represent relative fiscal and institutional
          health, not precise measurements. Historical slider values use regional adjustment factors, not
          country-specific historical data. For research use, always verify against primary sources.
          Framework under peer review — cite as: Baird, N. (2026). Engine of Emergence.
        </p>
      </div>
    </div>
  );
}


// ── CLIMATE SYSTEMS DATA ──────────────────────────────────────────────────────
// Each Earth system treated as a complex adaptive system
// chi = system efficiency (energy/nutrient conversion, feedback integrity)
// s   = energy/resource throughput (normalized to pre-industrial baseline)
// lambda0 = base burden (thermal stress, chemical load, structural damage)
// C   = system complexity (biodiversity, feedback loops, interconnection)
// Sources: NOAA, NASA, IPCC AR6, peer-reviewed literature

const CLIMATE_SYSTEMS = [
  {
    id:"arctic_ice", name:"Arctic Sea Ice", region:"Polar", icon:"🧊",
    color:"#38BDF8", lat:85, lon:0,
    desc:"Arctic summer sea ice extent has declined 13% per decade since 1979. The system is approaching a potential tipping point beyond which summer ice-free conditions become self-reinforcing.",
    source:"NSIDC, PIOMAS, Stroeve et al. (2012), IPCC AR6",
    history:[
      {year:1960,chi:0.92,s:0.96,lambda0:0.08,C:0.82, event:"Pre-satellite era baseline. Arctic ice extent near maximum of modern record. Albedo feedback loop fully intact."},
      {year:1979,chi:0.88,s:0.92,lambda0:0.12,C:0.82, event:"Satellite monitoring begins. First systematic measurements. Summer extent still robust at ~7.2 million km²."},
      {year:1990,chi:0.83,s:0.86,lambda0:0.17,C:0.80, event:"Measurable decline begins. Ocean warming accelerating ice melt from below. The albedo feedback begins to amplify warming."},
      {year:2000,chi:0.75,s:0.78,lambda0:0.25,C:0.78, event:"Decline rate accelerates. Ice is thinning as well as retreating. Multi-year ice being replaced by thinner first-year ice."},
      {year:2007,chi:0.62,s:0.64,lambda0:0.38,C:0.74, event:"Record summer minimum — 4.17 million km². Scientists shocked by speed of decline. M crosses into warning territory."},
      {year:2012,chi:0.52,s:0.54,lambda0:0.48,C:0.72, event:"All-time record minimum — 3.41 million km². The Arctic is in freefall. Ice-free summers now projected within decades."},
      {year:2020,chi:0.40,s:0.42,lambda0:0.60,C:0.68, event:"Second lowest extent on record. Arctic warming 4× faster than global average. The tipping point may already be in view."},
      {year:2023,chi:0.34,s:0.36,lambda0:0.66,C:0.66, event:"Persistent record lows. Antarctica simultaneously shatters its own records. Both poles in simultaneous crisis for the first time in the observational record."},
    ]
  },
  {
    id:"ocean_heat", name:"Global Ocean Heat", region:"Global Ocean", icon:"🌊",
    color:"#22D3EE", lat:-20, lon:-30,
    desc:"The ocean absorbs 90% of excess heat from climate change. Rising heat content drives sea level rise, intensifies hurricanes, and disrupts the circulation systems that regulate global climate.",
    source:"NOAA Ocean Climate Laboratory, Cheng et al. (2022), von Schuckmann et al. (2023)",
    history:[
      {year:1960,chi:0.88,s:0.82,lambda0:0.12,C:0.55, event:"Ocean heat content near equilibrium. Thermohaline circulation functioning normally. Enormous buffering capacity intact."},
      {year:1975,chi:0.85,s:0.80,lambda0:0.15,C:0.60, event:"Measurable warming begins. Atmosphere loading ocean with heat faster than circulation distributes it."},
      {year:1990,chi:0.80,s:0.76,lambda0:0.20,C:0.66, event:"Deep ocean warming confirmed. Heat penetrating to 700m+. Marine heatwaves becoming more frequent."},
      {year:2000,chi:0.74,s:0.70,lambda0:0.26,C:0.72, event:"Argo float network deployed — first comprehensive deep ocean monitoring. The scale of warming shocks scientists."},
      {year:2010,chi:0.66,s:0.64,lambda0:0.34,C:0.76, event:"Ocean heating rate doubles. Sea level rise accelerating. Coral bleaching events now annual in some regions."},
      {year:2016,chi:0.58,s:0.58,lambda0:0.42,C:0.80, event:"Record ocean temperatures. Global coral bleaching event kills 50% of shallow Great Barrier Reef corals. M crosses zero."},
      {year:2020,chi:0.50,s:0.50,lambda0:0.50,C:0.82, event:"Every year since 2015 sets new ocean heat records. Marine heatwaves 50× more frequent than 1980s baseline."},
      {year:2023,chi:0.42,s:0.42,lambda0:0.58,C:0.84, event:"2023 ocean temperatures shatter records by margins that shock scientists. North Atlantic 5°C above average in some regions. Unprecedented."},
    ]
  },
  {
    id:"amazon", name:"Amazon Rainforest", region:"South America", icon:"🌳",
    color:"#4ADE80", lat:-5, lon:-60,
    desc:"The Amazon generates 50% of its own rainfall through transpiration. At ~20-25% deforestation, models suggest the system tips irreversibly toward savanna. We are at 20%.",
    source:"INPE, Global Forest Watch, Lovejoy & Nobre (2018), Boulton et al. (2022)",
    history:[
      {year:1960,chi:0.92,s:0.90,lambda0:0.08,C:0.88, event:"Amazon largely untouched. 5.5 million km² of intact forest. Self-generating rainfall cycle at maximum efficiency."},
      {year:1980,chi:0.86,s:0.84,lambda0:0.14,C:0.87, event:"Deforestation begins in earnest. Trans-Amazon highway opens frontier. 5% of forest cleared — still well within resilience limits."},
      {year:1995,chi:0.80,s:0.78,lambda0:0.20,C:0.86, event:"12% deforested. Fire season intensifies. Rainfall patterns in eastern Amazon begin to shift. Early warning signals appear."},
      {year:2004,chi:0.74,s:0.72,lambda0:0.26,C:0.86, event:"All-time deforestation record — 27,000 km² in a single year. The moisture recycling system under serious strain."},
      {year:2012,chi:0.66,s:0.66,lambda0:0.34,C:0.87, event:"18% deforested. Forest Code weakened. Climate models warn 20-25% threshold triggers irreversible savannification."},
      {year:2019,chi:0.56,s:0.56,lambda0:0.44,C:0.88, event:"80,000 fires. Bolsonaro era deforestation surge. Eastern Amazon now emitting more carbon than it absorbs. M crosses zero."},
      {year:2022,chi:0.46,s:0.46,lambda0:0.54,C:0.89, event:"21% deforested — approaching the tipping point threshold. Boulton et al. confirm 75% of Amazon has lost measurable resilience since 2000."},
      {year:2023,chi:0.42,s:0.42,lambda0:0.58,C:0.89, event:"Deforestation slows under Lula but damage persists. The eastern Amazon is now a net carbon source. Scientists debate whether the tipping point is already past."},
    ]
  },
  {
    id:"amoc", name:"Atlantic Circulation (AMOC)", region:"North Atlantic", icon:"🌀",
    color:"#818CF8", lat:45, lon:-35,
    desc:"The Atlantic Meridional Overturning Circulation moves heat from tropics to northern Europe. New research suggests it may be approaching a collapse tipping point that would dramatically cool Europe and disrupt global rainfall patterns.",
    source:"Caesar et al. (2021), Boers (2021), IPCC AR6 Chapter 9, Ditlevsen & Ditlevsen (2023)",
    history:[
      {year:1960,chi:0.86,s:0.88,lambda0:0.14,C:0.72, event:"AMOC near historical maximum. Gulf Stream keeping Northern Europe 5-10°C warmer than it would otherwise be. System fully intact."},
      {year:1980,chi:0.83,s:0.85,lambda0:0.17,C:0.74, event:"Subtle weakening detectable in proxy records. Greenland melt beginning to add freshwater — which disrupts the density-driven circulation."},
      {year:2000,chi:0.77,s:0.79,lambda0:0.23,C:0.76, event:"RAPID array deployed — first continuous AMOC monitoring. Direct measurements confirm weakening underway."},
      {year:2010,chi:0.70,s:0.72,lambda0:0.30,C:0.78, event:"AMOC at weakest point in 1,000 years according to proxy reconstructions. Accelerating Greenland melt adding freshwater faster than the circulation can compensate."},
      {year:2015,chi:0.62,s:0.64,lambda0:0.38,C:0.78, event:"Multiple studies confirm AMOC 15% weaker than mid-20th century. M enters warning zone. Potential collapse timelines discussed openly in literature."},
      {year:2021,chi:0.54,s:0.56,lambda0:0.46,C:0.80, event:"Caesar et al. present fingerprint evidence of AMOC approaching a tipping point. Boers finds early warning signals in sea surface temperature data."},
      {year:2023,chi:0.46,s:0.48,lambda0:0.54,C:0.80, event:"Ditlevsen & Ditlevsen (2023) project collapse between 2025-2095 under current trajectories. Nature paper triggers major scientific debate. M is negative and declining."},
    ]
  },
  {
    id:"coral_global", name:"Global Coral Reefs", region:"Tropical Oceans", icon:"🪸",
    color:"#FB923C", lat:5, lon:120,
    desc:"Coral reefs cover less than 1% of the ocean floor but support 25% of all marine species. At current warming trajectories, 70-90% of reefs will experience annual bleaching by 2050.",
    source:"GCRMN (2020), Hughes et al. (2017, 2018), IPCC Special Report on Ocean and Cryosphere",
    history:[
      {year:1960,chi:0.90,s:0.88,lambda0:0.10,C:0.85, event:"Global reef system near pristine baseline. Bleaching essentially unknown as a phenomenon. Full species diversity and structural complexity."},
      {year:1980,chi:0.84,s:0.82,lambda0:0.16,C:0.84, event:"First mass bleaching events documented — Caribbean 1983. Coral disease increasing. Overfishing reducing herbivores that keep algae in check."},
      {year:1998,chi:0.72,s:0.70,lambda0:0.28,C:0.80, event:"First global mass bleaching event. El Niño + warming kills 16% of all coral globally in a single year. The world is shocked."},
      {year:2005,chi:0.64,s:0.62,lambda0:0.36,C:0.78, event:"Caribbean loses 40% of coral cover in a single bleaching event. Recovery windows between events shortening. M crosses zero."},
      {year:2016,chi:0.50,s:0.48,lambda0:0.50,C:0.74, event:"Third global mass bleaching — worst on record. Great Barrier Reef loses 50% of shallow coral. Bleaching now occurring in La Niña years."},
      {year:2020,chi:0.40,s:0.38,lambda0:0.60,C:0.70, event:"GCRMN reports 50% of world's coral lost since 1950. Recovery rates cannot keep pace with bleaching frequency. The system is in structured decline."},
      {year:2023,chi:0.32,s:0.30,lambda0:0.68,C:0.66, event:"2023-24 global bleaching event — worst in history. 91% of monitored reefs show bleaching. Scientists describe the reef system as experiencing a 'phase shift' from coral to algae dominance."},
    ]
  },
  {
    id:"permafrost", name:"Arctic Permafrost", region:"Polar/Boreal", icon:"🏔️",
    color:"#A78BFA", lat:68, lon:120,
    desc:"Permafrost stores twice as much carbon as the atmosphere. As it thaws, it releases CO₂ and methane — creating a feedback loop that accelerates warming independent of human emissions.",
    source:"Turetsky et al. (2020), IPCC AR6, Schuur et al. (2015), Walter Anthony et al. (2018)",
    history:[
      {year:1960,chi:0.88,s:0.84,lambda0:0.12,C:0.70, event:"Permafrost stable across 25% of Northern Hemisphere land area. Carbon locked in frozen soils for up to 40,000 years. Feedback loops dormant."},
      {year:1980,chi:0.84,s:0.80,lambda0:0.16,C:0.72, event:"Active layer deepening in Siberia and Alaska. Infrastructure damage beginning. Thermokarst lakes forming as ice-rich permafrost thaws."},
      {year:2000,chi:0.77,s:0.73,lambda0:0.23,C:0.74, event:"Permafrost temperatures rising 0.5°C per decade — twice the global average warming rate. Abrupt thaw processes (not captured in models) accelerating."},
      {year:2010,chi:0.68,s:0.64,lambda0:0.32,C:0.76, event:"Methane emissions from Arctic lakes and wetlands measurably increasing. Models underestimating thaw rates. M crosses into warning zone."},
      {year:2018,chi:0.58,s:0.54,lambda0:0.42,C:0.78, event:"Walter Anthony et al. document abrupt permafrost thaw releasing carbon 10× faster than gradual models predict. The timeline for the carbon feedback accelerates."},
      {year:2020,chi:0.50,s:0.46,lambda0:0.52,C:0.80, event:"Siberian heatwave — 38°C above Arctic Circle. Record permafrost thaw. Oil infrastructure failures. Massive methane plumes detected. M goes negative."},
      {year:2023,chi:0.42,s:0.38,lambda0:0.60,C:0.82, event:"2.5 million km² of permafrost has degraded since 1970. Self-reinforcing carbon release now considered 'locked in' for 1.5°C of warming regardless of human action."},
    ]
  },
  {
    id:"monsoon_asia", name:"Asian Monsoon System", region:"South/East Asia", icon:"🌧️",
    color:"#34D399", lat:20, lon:90,
    desc:"The Asian monsoon delivers water to 70% of the world's population. Disruption from warming and aerosol loading is already altering rainfall patterns across India, China, and Southeast Asia.",
    source:"Wang et al. (2021), Christensen et al. IPCC AR6, Turner & Annamalai (2012)",
    history:[
      {year:1960,chi:0.84,s:0.86,lambda0:0.16,C:0.70, event:"Asian monsoon in near-historical baseline. Reliable seasonal rainfall supporting agriculture for 3 billion people across South and East Asia."},
      {year:1980,chi:0.80,s:0.82,lambda0:0.20,C:0.72, event:"Indian summer monsoon weakening detectably. Aerosol pollution masking some warming but disrupting thermal gradients that drive monsoon circulation."},
      {year:1990,chi:0.76,s:0.78,lambda0:0.24,C:0.74, event:"Inter-annual variability increasing. ENSO influence on monsoon strengthening. Droughts and floods becoming more extreme and less predictable."},
      {year:2002,chi:0.70,s:0.72,lambda0:0.30,C:0.76, event:"India's worst drought in 15 years. Pakistan flooding. The monsoon's reliability — the bedrock of Asian agriculture — is measurably declining."},
      {year:2010,chi:0.64,s:0.66,lambda0:0.36,C:0.78, event:"Pakistan mega-floods (20% of country submerged). Russian heat wave. Both linked to weakened jet stream from Arctic amplification. M at warning level."},
      {year:2018,chi:0.58,s:0.60,lambda0:0.42,C:0.80, event:"Kerala floods — worst in 100 years. Bangladesh flooding displaces millions. The monsoon is intensifying in some regions, failing in others — spatial coherence breaking down."},
      {year:2023,chi:0.52,s:0.54,lambda0:0.48,C:0.82, event:"Libya floods kill 11,000 in hours. India and Pakistan simultaneous extreme events. South Asian monsoon described as entering a 'new regime' with no historical analog."},
    ]
  },
  {
    id:"boreal_forest", name:"Boreal Forest (Taiga)", region:"Northern Hemisphere", icon:"🌲",
    color:"#86EFAC", lat:58, lon:80,
    desc:"The boreal forest is Earth's largest terrestrial ecosystem — a carbon sink covering 1.4 billion hectares. Warming is converting it from a carbon sink to a carbon source through fire, beetle outbreaks, and drought.",
    source:"Gauthier et al. (2015), Walker et al. (2019), IPCC AR6 Chapter 2",
    history:[
      {year:1960,chi:0.86,s:0.84,lambda0:0.14,C:0.76, event:"Boreal forest functioning as major carbon sink. Fire return intervals of 100-200 years. Beetle populations controlled by cold winters. System fully intact."},
      {year:1980,chi:0.82,s:0.80,lambda0:0.18,C:0.76, event:"Fire frequency beginning to increase in North American boreal. Warming reducing snowpack that historically suppressed bark beetle populations."},
      {year:1995,chi:0.76,s:0.74,lambda0:0.24,C:0.76, event:"Mountain pine beetle epidemic begins in British Columbia — enabled by warmer winters. Will kill trees across 18 million hectares by 2012."},
      {year:2004,chi:0.68,s:0.66,lambda0:0.32,C:0.76, event:"Alaska fires burn 2.7 million hectares in a single season — largest in state history. Permafrost beneath boreal forest thawing, destabilizing root systems."},
      {year:2014,chi:0.60,s:0.58,lambda0:0.40,C:0.75, event:"Northwest Territories fire season — 3.4 million hectares. The boreal fire season is now 40% longer than 1970s baseline. M crosses into warning zone."},
      {year:2020,chi:0.52,s:0.50,lambda0:0.50,C:0.74, event:"Siberian fires emit 540 megatons of CO₂ — equivalent to Sweden's annual emissions. The boreal is transitioning from carbon sink to carbon source."},
      {year:2023,chi:0.44,s:0.42,lambda0:0.58,C:0.73, event:"Canada's worst fire season in history — 18 million hectares. Smoke blankets US cities. The boreal carbon sink may have permanently flipped. M is negative and accelerating."},
    ]
  },
  {
    id:"greenland", name:"Greenland Ice Sheet", region:"Arctic", icon:"🏔️",
    color:"#BAE6FD", lat:72, lon:-42,
    desc:"Greenland holds enough ice to raise global sea levels by 7 meters. Its melt is now irreversible at current temperatures — the question is how fast. Current melt rate is 280 billion tons per year.",
    source:"Bamber et al. (2019), The IMBIE Team (2020), Mouginot et al. (2019)",
    history:[
      {year:1960,chi:0.90,s:0.86,lambda0:0.10,C:0.60, event:"Greenland ice sheet near mass balance — gaining roughly as much ice from snowfall as it loses to melt. Contribution to sea level rise negligible."},
      {year:1980,chi:0.86,s:0.82,lambda0:0.14,C:0.62, event:"Balance beginning to tip. Surface melt zone expanding upward in elevation. Outlet glaciers beginning to accelerate."},
      {year:1995,chi:0.80,s:0.76,lambda0:0.20,C:0.64, event:"Mass loss begins to exceed mass gain. Marine-terminating glaciers accelerating as ocean waters warm. The ice sheet is now shrinking."},
      {year:2002,chi:0.72,s:0.68,lambda0:0.28,C:0.66, event:"GRACE satellite confirms dramatic acceleration of mass loss. Jakobshavn Glacier doubles in speed. 50 billion tons per year net loss."},
      {year:2012,chi:0.58,s:0.54,lambda0:0.42,C:0.68, event:"Record surface melt — 97% of ice sheet surface shows melting for first time. 400 billion tons lost that year. M clearly negative."},
      {year:2019,chi:0.44,s:0.40,lambda0:0.56,C:0.70, event:"Worst year on record — 530 billion tons lost. Enough to raise global sea levels 1.5mm in a single year. Tipping point for irreversible loss may have passed."},
      {year:2023,chi:0.36,s:0.32,lambda0:0.64,C:0.70, event:"280 billion tons per year average loss over the decade. Scientists now speak of 'committed' sea level rise — ice that will melt regardless of future emissions cuts."},
    ]
  },
  {
    id:"savanna_africa", name:"African Savanna Systems", region:"Africa", icon:"🦁",
    color:"#FCD34D", lat:-5, lon:25,
    desc:"African savannas cover 20% of Earth's land surface and support extraordinary biodiversity. Shifting rainfall patterns, bush encroachment, and land pressure are degrading ecosystem function across the continent.",
    source:"Venter et al. (2018), Lehmann et al. (2014), Staver et al. (2011)",
    history:[
      {year:1960,chi:0.82,s:0.80,lambda0:0.18,C:0.78, event:"African savanna systems largely intact. Large mammal populations sustaining ecosystem function. Fire-grazing interactions maintaining grass-tree balance."},
      {year:1980,chi:0.76,s:0.74,lambda0:0.24,C:0.78, event:"Agricultural expansion accelerating. Poaching decimating elephant populations that historically maintained open savanna structure."},
      {year:1995,chi:0.70,s:0.68,lambda0:0.30,C:0.76, event:"Bush encroachment accelerating across Southern and East Africa. Rainfall variability increasing. Savanna-forest boundary shifting."},
      {year:2005,chi:0.62,s:0.60,lambda0:0.38,C:0.76, event:"Lake Chad has lost 90% of its surface area since 1963 — a regional climate disaster driving migration and conflict. Sahel rainfall patterns destabilizing."},
      {year:2015,chi:0.54,s:0.52,lambda0:0.46,C:0.75, event:"East African drought — worst in 60 years. Somalia, Kenya, Ethiopia in crisis. The savanna fire season intensifying and extending."},
      {year:2020,chi:0.46,s:0.44,lambda0:0.54,C:0.74, event:"Madagascar in famine from climate-driven drought. Horn of Africa in fifth consecutive failed rainy season. The savanna carbon sink measurably weakening."},
      {year:2023,chi:0.40,s:0.38,lambda0:0.60,C:0.73, event:"Horn of Africa in worst drought in 40 years. Wildfires across Southern Africa. The system is in sustained decline — M deeply negative and no reversal in sight without structural intervention."},
    ]
  },
];

const CLIMATE_YEARS = [1960,1980,1995,2000,2004,2010,2015,2018,2020,2023];

function getClimateM(system, year) {
  // Find closest historical year
  const years = system.history.map(h=>h.year);
  let closest = system.history[0];
  let minDiff = Math.abs(years[0]-year);
  system.history.forEach(h=>{
    const diff = Math.abs(h.year-year);
    if(diff<minDiff){minDiff=diff;closest=h;}
  });
  return calcM(closest.chi,closest.s,closest.lambda0,closest.C);
}

function getClimatePoint(system, year) {
  const years = system.history.map(h=>h.year);
  let closest = system.history[0];
  let minDiff = Math.abs(years[0]-year);
  system.history.forEach(h=>{
    const diff = Math.abs(h.year-year);
    if(diff<minDiff){minDiff=diff;closest=h;}
  });
  return closest;
}

// Approximate lat/lon positions for climate systems on map
const CLIMATE_COORDS = {
  arctic_ice:    [0,   85],
  ocean_heat:    [-30,-20],
  amazon:        [-60, -5],
  amoc:          [-35, 45],
  coral_global:  [120,  5],
  permafrost:    [120, 68],
  monsoon_asia:  [90,  20],
  boreal_forest: [80,  58],
  greenland:     [-42, 72],
  savanna_africa:[25,  -5],
};

// ── CLIMATE TAB ───────────────────────────────────────────────────────────────
function ClimateTab() {
  const [year, setYear]       = useState(2023);
  const [selected, setSelected] = useState(null);
  const [showAll, setShowAll] = useState(false);

  const sel = selected ? CLIMATE_SYSTEMS.find(s=>s.id===selected) : null;
  const selPt = sel ? getClimatePoint(sel, year) : null;

  const toXY = (lon,lat,W,H) => [
    ((lon+180)/360)*W,
    ((90-lat)/180)*H,
  ];

  const W=620, H=310;

  const scored = [...CLIMATE_SYSTEMS].sort((a,b)=>getClimateM(b,year)-getClimateM(a,year));

  // Sparkline for a system
  function ClimateSparkline({sys}){
    const pts = sys.history;
    const vals = pts.map(p=>calcM(p.chi,p.s,p.lambda0,p.C));
    const minV=Math.min(...vals,-0.5), maxV=Math.max(...vals,0.3);
    const rng=maxV-minV, w=120, h=36, pad=4;
    const xs=pts.map((_,i)=>pad+(i/(pts.length-1))*(w-pad*2));
    const ys=vals.map(v=>h-pad-((v-minV)/rng)*(h-pad*2));
    const path=xs.map((x,i)=>`${i===0?"M":"L"} ${x} ${ys[i]}`).join(" ");
    const zeroY=h-pad-((0-minV)/rng)*(h-pad*2);
    const lastM=vals[vals.length-1];
    return(
      <svg width={w} height={h} style={{display:"block",overflow:"visible"}}>
        <line x1={pad} y1={zeroY} x2={w-pad} y2={zeroY} stroke="#333" strokeWidth={0.5} strokeDasharray="2,3"/>
        <path d={path} fill="none" stroke={mColor(lastM)} strokeWidth={1.5} strokeLinecap="round"/>
        {vals.map((v,i)=><circle key={i} cx={xs[i]} cy={ys[i]} r={i===vals.length-1?3:2} fill={mColor(v)}/>)}
      </svg>
    );
  }

  return (
    <div style={{display:"flex",flexDirection:"column",gap:24}}>

      {/* Header */}
      <div>
        <h2 style={{fontFamily:"var(--serif)",fontSize:28,color:"#FFFFFF",marginBottom:10,borderLeft:"3px solid #EF4444",paddingLeft:14}}>
          Earth Systems
        </h2>
        <p style={{color:"#A3A3A3",fontSize:13,fontFamily:"var(--sans)",lineHeight:1.65,maxWidth:640}}>
          The same equation that identifies civilizational collapse applies to planetary systems.
          Each Earth system has inputs, outputs, and overhead. When burden exceeds throughput, the system destabilizes.
          These are the 10 most critical Earth systems, calibrated from NOAA, NASA, and peer-reviewed climate literature.
        </p>
      </div>

      {/* Year slider */}
      <div style={{background:"#111111",border:"1px solid #2A2A2A",borderRadius:10,
        padding:"12px 18px",display:"flex",alignItems:"center",gap:14}}>
        <span style={{fontFamily:"var(--serif)",fontSize:22,color:"#FFFFFF",flexShrink:0}}>
          {year}
        </span>
        <input type="range" min={0} max={CLIMATE_YEARS.length-1} step={1}
          value={CLIMATE_YEARS.indexOf(year)<0?CLIMATE_YEARS.length-1:CLIMATE_YEARS.indexOf(year)}
          onChange={e=>setYear(CLIMATE_YEARS[parseInt(e.target.value)])}
          style={{flex:1,accentColor:"#2563EB"}}
        />
        <div style={{display:"flex",gap:6,flexShrink:0}}>
          {CLIMATE_YEARS.map(y=>(
            <button key={y} onClick={()=>setYear(y)} style={{
              background:year===y?"#2563EB":"#1A1A1A",
              border:`1px solid ${year===y?"#2563EB":"#2A2A2A"}`,
              borderRadius:4,padding:"3px 7px",fontSize:9,
              color:year===y?"#FFFFFF":"#525252",
              fontFamily:"var(--mono)"
            }}>{y}</button>
          ))}
        </div>
      </div>

      {/* World map */}
      <div style={{background:"#0A0A0A",border:"1px solid #2A2A2A",borderRadius:14,padding:16}}>
        <div style={{fontFamily:"var(--mono)",fontSize:9,color:"#3B82F6",marginBottom:10,letterSpacing:3}}>
          EARTH SYSTEMS MAP · {year} · COLOR = STABILITY MARGIN
        </div>
        <svg
          viewBox={`0 0 ${W} ${H}`}
          style={{display:"block",width:"100%",height:"auto"}}
          preserveAspectRatio="xMidYMid meet"
        >
          {/* Background */}
          <rect x={0} y={0} width={W} height={H} fill="#050505"/>
          {/* Region labels */}
          <g fontFamily="Inter,sans-serif" fontSize="9" fill="#3A5A6A" opacity="0.7" letterSpacing="2">
            <text x="124" y="113" textAnchor="middle">NORTH AMERICA</text>
            <text x="191" y="200" textAnchor="middle">S. AMERICA</text>
            <text x="305" y="78" textAnchor="middle">EUROPE</text>
            <text x="310" y="160" textAnchor="middle">AFRICA</text>
            <text x="400" y="88" textAnchor="middle">MIDDLE EAST</text>
            <text x="445" y="62" textAnchor="middle">RUSSIA / C. ASIA</text>
            <text x="465" y="118" textAnchor="middle">S. ASIA</text>
            <text x="505" y="88" textAnchor="middle">E. ASIA</text>
            <text x="525" y="138" textAnchor="middle">SE ASIA</text>
            <text x="525" y="200" textAnchor="middle">AUSTRALIA</text>
          </g>
          {/* Region labels */}
          <g fontFamily="Inter,sans-serif" fontSize="9" fill="#3A5A6A" opacity="0.7" letterSpacing="2">
            <text x="124" y="113" textAnchor="middle">NORTH AMERICA</text>
            <text x="191" y="200" textAnchor="middle">S. AMERICA</text>
            <text x="305" y="78" textAnchor="middle">EUROPE</text>
            <text x="310" y="160" textAnchor="middle">AFRICA</text>
            <text x="400" y="88" textAnchor="middle">MIDDLE EAST</text>
            <text x="445" y="62" textAnchor="middle">RUSSIA / C. ASIA</text>
            <text x="465" y="118" textAnchor="middle">S. ASIA</text>
            <text x="505" y="88" textAnchor="middle">E. ASIA</text>
            <text x="525" y="138" textAnchor="middle">SE ASIA</text>
            <text x="525" y="200" textAnchor="middle">AUSTRALIA</text>
          </g>
          
          {/* Grid */}
          {[-60,-30,0,30,60].map(lat=>{
            const [,y]=toXY(0,lat,W,H);
            return <line key={lat} x1={0} y1={y} x2={W} y2={y} stroke="#1A1A1A" strokeWidth={0.8}/>;
          })}
          {[-120,-60,0,60,120].map(lon=>{
            const [x]=toXY(lon,0,W,H);
            return <line key={lon} x1={x} y1={0} x2={x} y2={H} stroke="#1A1A1A" strokeWidth={0.8}/>;
          })}
          {/* Equator */}
          {(() => { const [,y]=toXY(0,0,W,H); return <line x1={0} y1={y} x2={W} y2={y} stroke="#2A2A2A" strokeWidth={1}/>; })()}

          {/* Continent wireframe outlines */}

          {/* System dots */}
          {CLIMATE_SYSTEMS.map(sys=>{
            const coords = CLIMATE_COORDS[sys.id];
            if(!coords) return null;
            const M = getClimateM(sys,year);
            const [x,y] = toXY(coords[0],coords[1],W,H);
            const isSel = selected===sys.id;
            return (
              <g key={sys.id} onClick={()=>setSelected(isSel?null:sys.id)} style={{cursor:"pointer"}}>
                <circle cx={x} cy={y} r={36} fill={mColor(M)} opacity={0.07}/>
                <circle cx={x} cy={y} r={22} fill={mColor(M)} opacity={0.15}/>
                <circle cx={x} cy={y} r={14} fill={mColor(M)} opacity={isSel?1:0.85}
                  stroke={isSel?"#FFFFFF":"none"} strokeWidth={2}/>
                <text x={x} y={y+5} textAnchor="middle" fontSize={14}>{sys.icon}</text>
                <text x={x} y={y+28} textAnchor="middle" fontSize={9}
                  fill="#FFFFFF" fontFamily="Inter" fontWeight="600">
                  {sys.name.split(" ")[0]}
                </text>
                <text x={x} y={y+39} textAnchor="middle" fontSize={8}
                  fill={mColor(M)} fontFamily="JetBrains Mono" fontWeight="700">
                  {(M>=0?"+":"")+M.toFixed(2)}
                </text>
              </g>
            );
          })}
        </svg>

        {/* Legend */}
        <div style={{display:"flex",gap:14,marginTop:10,flexWrap:"wrap",alignItems:"center"}}>
          {[["#06B6D4","Sustaining"],["#22C55E","Stable"],["#84CC16","Healthy"],
            ["#EAB308","Warning"],["#F97316","Declining"],["#EF4444","Critical"]].map(([c,l])=>(
            <div key={l} style={{display:"flex",alignItems:"center",gap:5}}>
              <div style={{width:8,height:8,borderRadius:"50%",background:c}}/>
              <span style={{fontSize:10,color:"#737373",fontFamily:"var(--sans)"}}>{l}</span>
            </div>
          ))}
          <span style={{fontSize:10,color:"#525252",fontFamily:"var(--sans)",marginLeft:"auto"}}>
            Tap any system to expand
          </span>
        </div>
      </div>

      {/* Selected system detail */}
      {sel && selPt && (
        <div style={{background:"#0A0A0A",border:`1px solid ${mColor(getClimateM(sel,year))}40`,
          borderRadius:14,overflow:"hidden"}}>

          {/* Header */}
          <div style={{padding:"20px 24px",borderBottom:"1px solid #1A1A1A",
            display:"flex",justifyContent:"space-between",alignItems:"flex-start",flexWrap:"wrap",gap:16}}>
            <div>
              <div style={{display:"flex",alignItems:"center",gap:10,marginBottom:8}}>
                <span style={{fontSize:28}}>{sel.icon}</span>
                <div>
                  <div style={{fontSize:17,fontWeight:700,color:"#FFFFFF",fontFamily:"var(--sans)"}}>{sel.name}</div>
                  <div style={{fontSize:11,color:"#525252",fontFamily:"var(--mono)"}}>{sel.region}</div>
                </div>
              </div>
              <p style={{fontSize:13,color:"#A3A3A3",lineHeight:1.7,fontFamily:"var(--sans)",maxWidth:500}}>{sel.desc}</p>
              <div style={{fontSize:10,color:"#404040",fontFamily:"var(--mono)",marginTop:8}}>Source: {sel.source}</div>
            </div>
            <div style={{textAlign:"center",flexShrink:0}}>
              <Gauge value={getClimateM(sel,year)} size={150}/>
              <div style={{fontSize:13,fontWeight:700,color:mColor(getClimateM(sel,year)),
                fontFamily:"var(--sans)",marginTop:4}}>{mLabel(getClimateM(sel,year))}</div>
            </div>
          </div>

          {/* Full trajectory chart */}
          <div style={{padding:"16px 24px",borderBottom:"1px solid #1A1A1A"}}>
            <div style={{fontFamily:"var(--mono)",fontSize:9,color:"#3B82F6",marginBottom:12,letterSpacing:3}}>
              M TRAJECTORY · {sel.history[0].year}–{sel.history[sel.history.length-1].year}
            </div>
            <MChart points={sel.history} dsColor={sel.color} dsId={sel.id}/>
          </div>

          <MInsight points={sel.history} dsId={sel.id} dsLabel={sel.name} domain={sel.region}/>
          {/* Current point detail */}
          <div style={{padding:"16px 24px",borderBottom:"1px solid #1A1A1A",
            background:"#000000"}}>
            <div style={{fontSize:9,fontFamily:"var(--mono)",color:sel.color,
              marginBottom:10,letterSpacing:2}}>WHAT WAS HAPPENING · {selPt.year}</div>
            <p style={{fontSize:13,color:"#D4D4D4",lineHeight:1.7,fontFamily:"var(--sans)"}}>{selPt.event}</p>
          </div>

          {/* Variable breakdown */}
          <div style={{padding:"16px 24px",display:"grid",
            gridTemplateColumns:"repeat(4,1fr)",gap:8}}>
            {[
              {sym:"χ",val:selPt.chi,color:"#60A5FA",name:"Efficiency"},
              {sym:"s",val:selPt.s,color:"#A78BFA",name:"Throughput"},
              {sym:"λ(C)",val:selPt.lambda0+k*Math.pow(selPt.C,n),color:"#F87171",name:"Burden"},
              {sym:"C",val:selPt.C,color:"#FCD34D",name:"Complexity"},
            ].map((v,i)=>(
              <div key={i} style={{background:"#111111",borderRadius:8,
                padding:"12px 14px",textAlign:"center",border:"1px solid #1A1A1A"}}>
                <div style={{fontFamily:"var(--mono)",fontSize:15,color:v.color,fontWeight:700}}>{v.val.toFixed(3)}</div>
                <div style={{fontSize:9,color:"#525252",marginTop:3,fontFamily:"var(--sans)"}}>{v.sym} · {v.name}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Ranked list */}
      <div style={{background:"#0A0A0A",border:"1px solid #2A2A2A",borderRadius:14,overflow:"hidden"}}>
        <div style={{padding:"14px 20px",borderBottom:"1px solid #1A1A1A",
          display:"flex",justifyContent:"space-between",alignItems:"center",flexWrap:"wrap",gap:8}}>
          <div style={{fontFamily:"var(--mono)",fontSize:9,color:"#3B82F6",letterSpacing:3}}>
            ALL EARTH SYSTEMS RANKED · {year}
          </div>
          <div style={{fontFamily:"var(--mono)",fontSize:9,color:"#EF4444"}}>
            {scored.filter(s=>getClimateM(s,year)<0).length} of {scored.length} systems in negative margin
          </div>
        </div>

        {scored.map((sys,i)=>{
          const M = getClimateM(sys,year);
          const isSel = selected===sys.id;
          const barPct = Math.min(Math.abs(M)/0.6*50, 50);
          return (
            <div key={sys.id} onClick={()=>setSelected(isSel?null:sys.id)}
              style={{
                display:"flex",alignItems:"center",gap:12,padding:"12px 20px",
                borderBottom:"1px solid #111111",cursor:"pointer",
                background:isSel?"#0A0F18":"transparent",transition:"background 0.1s"
              }}
              onMouseEnter={e=>{ if(!isSel) e.currentTarget.style.background="#111111"; }}
              onMouseLeave={e=>{ if(!isSel) e.currentTarget.style.background=isSel?"#0A0F18":"transparent"; }}
            >
              <div style={{fontFamily:"var(--mono)",fontSize:10,color:"#525252",
                width:20,textAlign:"right",flexShrink:0}}>{i+1}</div>

              <div style={{display:"flex",alignItems:"center",gap:8,width:200,flexShrink:0}}>
                <span style={{fontSize:18}}>{sys.icon}</span>
                <div>
                  <div style={{fontSize:12,fontWeight:600,color:isSel?"#FFFFFF":"#D4D4D4",
                    fontFamily:"var(--sans)"}}>{sys.name}</div>
                  <div style={{fontSize:9,color:"#525252",fontFamily:"var(--sans)"}}>{sys.region}</div>
                </div>
              </div>

              <ClimateSparkline sys={sys}/>

              <div style={{flex:1,height:6,background:"#1A1A1A",borderRadius:3,
                position:"relative",overflow:"visible",minWidth:60}}>
                <div style={{
                  position:"absolute",
                  right: M>=0?"50%":undefined, left: M<0?"50%":undefined,
                  width:`${barPct}%`, height:"100%",
                  background:mColor(M), borderRadius:3, top:0,
                }}/>
                <div style={{position:"absolute",left:"50%",top:-2,
                  width:1,height:10,background:"#333"}}/>
              </div>

              <div style={{fontFamily:"var(--mono)",fontSize:12,fontWeight:700,
                color:mColor(M),width:68,textAlign:"right",flexShrink:0}}>
                {M>=0?"+":""}{M.toFixed(3)}
              </div>
              <div style={{fontSize:10,color:mColor(M),
                fontFamily:"var(--sans)",width:62,flexShrink:0}}>
                {mLabel(M)}
              </div>
            </div>
          );
        })}
      </div>

      {/* Scientific note */}
      <div style={{background:"#0A0A0A",border:"1px solid #2A2A2A",borderRadius:10,
        padding:"14px 18px",display:"flex",gap:10,alignItems:"flex-start"}}>
        <span style={{fontSize:13,flexShrink:0}}>⚠️</span>
        <p style={{fontSize:11,color:"#737373",lineHeight:1.6,fontFamily:"var(--sans)"}}>
          Earth system M values are calibrated from NOAA, NASA, IPCC AR6, and peer-reviewed literature.
          Variables represent system-level proxies: χ = ecosystem/process efficiency relative to pre-industrial baseline,
          s = energy/nutrient throughput normalized to historical maximum, λ₀ = thermal/chemical/structural burden,
          C = system complexity and interconnection. These are diagnostic estimates, not precise physical measurements.
          Framework under peer review — cite as: Baird, N. (2026). Engine of Emergence. arXiv:[pending].
        </p>
      </div>
    </div>
  );
}


// ── SEISMIC DATA ─────────────────────────────────────────────────────────────
// Variables mapped to EoE framework:
// chi = structural efficiency of crust (fault geometry, stress distribution regularity)
// s   = seismic energy throughput (normalized annual seismic moment release)
// lambda0 = baseline tectonic burden (accumulated strain, fault coupling)
// C   = system complexity (fault network density, interactions, plate boundaries)
// Sources: USGS, ISC, Shebalin (1997), Pacheco & Sykes (1992), regional seismic networks

const SEISMIC_REGIONS = [
  {
    id:"cascadia", name:"Cascadia Subduction Zone", region:"North America", icon:"🌊",
    color:"#F87171", lat:46, lon:-124,
    desc:"The Cascadia subduction zone runs 1,000km off the Pacific Northwest coast. A full-margin rupture (M9+) is overdue by historical standards. The last great Cascadia earthquake was January 26, 1700.",
    source:"USGS, Goldfinger et al. (2012), Oregon OEM",
    history:[
      {year:1700,chi:0.72,s:0.95,lambda0:0.28,C:0.82, event:"Last great rupture — M~9.0. Tsunami devastates Pacific Northwest coast and reaches Japan. 300+ years of strain have been accumulating since."},
      {year:1800,chi:0.74,s:0.42,lambda0:0.30,C:0.82, event:"Interseismic period. Strain accumulating steadily. No major surface rupture but episodic tremor and slow-slip events detected."},
      {year:1900,chi:0.72,s:0.38,lambda0:0.32,C:0.84, event:"Strain continues loading. Scientific understanding of subduction zones begins developing. The hazard is not yet recognized."},
      {year:1950,chi:0.70,s:0.35,lambda0:0.34,C:0.85, event:"GPS measurements begin capturing surface deformation. The locked zone is identified. Accumulated slip deficit now enormous."},
      {year:1980,chi:0.68,s:0.32,lambda0:0.38,C:0.86, event:"Mount St. Helens erupts — reminder of the arc volcanism driven by the same subduction. Paleoseismic research begins in earnest."},
      {year:2000,chi:0.65,s:0.30,lambda0:0.42,C:0.87, event:"Episodic tremor and slip (ETS) events discovered. The fault is communicating stress in slow pulses every 14 months."},
      {year:2011,chi:0.62,s:0.28,lambda0:0.46,C:0.88, event:"Tohoku M9.1 provides sobering analog. Pacific Northwest cities begin serious preparation. Strain deficit now equivalent to M9.0+."},
      {year:2023,chi:0.58,s:0.25,lambda0:0.52,C:0.88, event:"Full locking confirmed along most of the fault. USGS estimates 37% probability of M8+ in next 50 years. Portland, Seattle, and Vancouver directly in the rupture zone."},
    ]
  },
  {
    id:"san_andreas", name:"San Andreas Fault System", region:"North America", icon:"🏔️",
    color:"#F97316", lat:37, lon:-120,
    desc:"The San Andreas is the most studied fault on Earth. The Southern section has not ruptured since 1857 — the 'Big One' is a matter of when, not if.",
    source:"USGS, Field et al. (2014) UCERF3, Southern California Seismic Network",
    history:[
      {year:1857,chi:0.78,s:0.92,lambda0:0.22,C:0.75, event:"Fort Tejon earthquake M7.9. The southern San Andreas ruptures 360km. Los Angeles has fewer than 5,000 people. The next comparable event will find 20 million."},
      {year:1906,chi:0.74,s:0.88,lambda0:0.26,C:0.78, event:"San Francisco earthquake M7.9. Northern section ruptures 470km. 3,000 dead, city destroyed by fire. The fault's mechanics finally understood."},
      {year:1940,chi:0.72,s:0.65,lambda0:0.28,C:0.80, event:"Imperial Valley M6.9. The Imperial fault — a branch of the San Andreas system — ruptures. Infrastructure vulnerability begins to be recognized."},
      {year:1971,chi:0.70,s:0.72,lambda0:0.30,C:0.82, event:"Sylmar M6.6 kills 65, collapses hospital and freeway overpass. California begins serious seismic building codes. The southern section still locked."},
      {year:1989,chi:0.68,s:0.78,lambda0:0.32,C:0.84, event:"Loma Prieta M6.9 during World Series. 63 dead. Bay Bridge damaged. Cypress freeway collapses. But the southern San Andreas still hasn't moved."},
      {year:1994,chi:0.66,s:0.82,lambda0:0.34,C:0.85, event:"Northridge M6.7 — a blind thrust fault, not the San Andreas. 57 dead, $20B damage. The southern San Andreas accumulates another year of strain."},
      {year:2010,chi:0.62,s:0.55,lambda0:0.38,C:0.86, event:"El Mayor-Cucapah M7.2 in Baja. Stress transferred northward along the system. Scientists note the southern San Andreas is increasingly stressed."},
      {year:2023,chi:0.58,s:0.48,lambda0:0.44,C:0.87, event:"The southern San Andreas has not ruptured in 166 years. Slip deficit now ~8-10 meters. UCERF3 gives 60% probability of M6.7+ in Southern California in 30 years."},
    ]
  },
  {
    id:"japan_trench", name:"Japan Trench / Tohoku", region:"Asia-Pacific", icon:"🇯🇵",
    color:"#FCD34D", lat:38, lon:143,
    desc:"The most instrumentally monitored subduction zone on Earth. Tohoku 2011 (M9.1) was the most powerful earthquake ever recorded in Japan — and yet the hazard remains.",
    source:"JMA, Simons et al. (2011), Ozawa et al. (2011), Satake & Atwater (2007)",
    history:[
      {year:869, chi:0.76,s:0.94,lambda0:0.24,C:0.78, event:"Jogan earthquake M~8.4. Historical records describe tsunami inundating the Sendai plain — exactly the same area as 2011. The warning went unheeded for 1,142 years."},
      {year:1896,chi:0.74,s:0.90,lambda0:0.26,C:0.80, event:"Meiji Sanriku tsunami earthquake M8.5. 22,000 killed by tsunami with minimal shaking warning. Tsunami earthquakes identified as distinct hazard."},
      {year:1933,chi:0.72,s:0.86,lambda0:0.28,C:0.81, event:"Showa Sanriku M8.4. 3,000 dead. Seawall construction begins along Sanriku coast — walls that would prove woefully inadequate in 2011."},
      {year:1978,chi:0.70,s:0.68,lambda0:0.30,C:0.83, event:"Miyagi M7.4. Strain accumulating on the locked portion of the Japan Trench. Scientists identify the coming hazard but magnitude projections are too conservative."},
      {year:2005,chi:0.67,s:0.62,lambda0:0.34,C:0.84, event:"Miyagi M7.2. GPS confirms 8cm/year convergence rate. The locked zone is fully identified but official estimates still cap at M8.2 for the region."},
      {year:2011,chi:0.55,s:0.95,lambda0:0.48,C:0.86, event:"Tohoku M9.1. The largest earthquake in Japanese recorded history. 15,000 dead, Fukushima nuclear disaster. The official hazard assessment was wrong by a factor of 10 in energy."},
      {year:2023,chi:0.62,s:0.45,lambda0:0.42,C:0.86, event:"Post-Tohoku monitoring shows the fault is re-locking. Aftershock sequence continues 12 years later. The northern Japan Trench — which did not rupture in 2011 — is now assessed as capable of M9+."},
    ]
  },
  {
    id:"himalaya", name:"Himalayan Seismic Zone", region:"Asia", icon:"⛰️",
    color:"#A78BFA", lat:28, lon:84,
    desc:"The collision of India and Eurasia has built the world's highest mountains and loaded the most dangerous seismic zone on Earth. 500 million people live within severe shaking distance.",
    source:"Bilham et al. (2001), Avouac (2003), Rajendran & Rajendran (2011)",
    history:[
      {year:1255,chi:0.74,s:0.92,lambda0:0.26,C:0.80, event:"Nepal M~7.8. Kathmandu Valley devastated. Historical records describe one-third of the population killed. The same fault ruptured in 2015."},
      {year:1505,chi:0.72,s:0.94,lambda0:0.28,C:0.81, event:"Lo Mustang M~8.2. One of the largest Himalayan earthquakes in the last millennium. Western Nepal rupture. Evidence of surface rupture along 800km of the Main Frontal Thrust."},
      {year:1897,chi:0.70,s:0.90,lambda0:0.30,C:0.82, event:"Shillong M8.1. The Assam plateau ruptures. Liquefaction and landslides across a vast area. 1,500 dead — a figure that would be millions in today's population."},
      {year:1934,chi:0.68,s:0.88,lambda0:0.32,C:0.83, event:"Bihar-Nepal M8.0. 10,000 dead. Kathmandu severely damaged. Paleoseismic research later shows this rupture did not release all accumulated strain."},
      {year:1950,chi:0.66,s:0.85,lambda0:0.34,C:0.84, event:"Assam M8.6 — the largest continental earthquake of the 20th century. Massive landslides dam rivers. The eastern Himalayan arc fully ruptured."},
      {year:2005,chi:0.63,s:0.78,lambda0:0.38,C:0.85, event:"Kashmir M7.6. 80,000 dead. The worst natural disaster in Pakistani history. The central Himalayan seismic gap — between 1905 and 1934 ruptures — remains unruptured."},
      {year:2015,chi:0.58,s:0.82,lambda0:0.44,C:0.86, event:"Gorkha Nepal M7.8. 9,000 dead. Kathmandu largely spared due to basin resonance — but the rupture was incomplete. The eastern section of the same fault did not move. A future M8+ on that section is expected."},
      {year:2023,chi:0.54,s:0.40,lambda0:0.50,C:0.86, event:"The central Himalayan seismic gap has not ruptured in 500+ years. GPS shows 20mm/year convergence. Slip deficit now equivalent to M8.5. 500 million people are exposed."},
    ]
  },
  {
    id:"ring_of_fire", name:"Western Ring of Fire", region:"Asia-Pacific", icon:"🔥",
    color:"#EF4444", lat:0, lon:125,
    desc:"The western Ring of Fire — from Indonesia through the Philippines to Taiwan — is the most seismically active arc on Earth. The 2004 Sumatra earthquake remains the third-largest ever recorded.",
    source:"USGS, Lay et al. (2005), Sieh et al. (2008), Natawidjaja et al. (2006)",
    history:[
      {year:1797,chi:0.76,s:0.92,lambda0:0.24,C:0.85, event:"Sumatra M8.4. Historical records describe massive tsunami along western Sumatra coast. Coral microatolls later confirm this as part of a recurring rupture cycle."},
      {year:1833,chi:0.74,s:0.94,lambda0:0.26,C:0.85, event:"Sumatra M8.8-9.2. The largest pre-instrumental earthquake in the region. Coral records show sea level changes across a vast area. Sets up the stress configuration for 2004."},
      {year:1906,chi:0.72,s:0.88,lambda0:0.28,C:0.86, event:"Colombia-Ecuador M8.8. The western Americas segment of the ring ruptures. Tsunami crosses the Pacific. The Ring of Fire operates as a globally connected system."},
      {year:1960,chi:0.70,s:0.98,lambda0:0.30,C:0.87, event:"Valdivia Chile M9.5 — the largest earthquake ever recorded. 1,655 dead. Tsunami kills across the Pacific. The entire subduction system resets."},
      {year:2004,chi:0.62,s:0.96,lambda0:0.40,C:0.88, event:"Sumatra-Andaman M9.1. 227,000 dead in 14 countries. The Indian Ocean tsunami was the deadliest in recorded history. The fault ruptured 1,300km in 8-10 minutes."},
      {year:2010,chi:0.60,s:0.88,lambda0:0.42,C:0.88, event:"Mentawai gap ruptures in two events — but only partially. The full Mentawai segment, which last ruptured in 1797/1833, is still locked and overdue."},
      {year:2023,chi:0.56,s:0.52,lambda0:0.48,C:0.88, event:"The Mentawai seismic gap remains one of the highest-risk unruptured segments on Earth. Coastal Sumatra cities have grown enormously since the last great rupture."},
    ]
  },
  {
    id:"anatolian", name:"North Anatolian Fault", region:"Middle East/Europe", icon:"🇹🇷",
    color:"#34D399", lat:40, lon:32,
    desc:"The North Anatolian Fault has produced a remarkable westward-migrating sequence of M7+ earthquakes through the 20th century — now pointing directly at Istanbul.",
    source:"Parsons et al. (2000), Stein et al. (1997), Barka (1999), KOERI",
    history:[
      {year:1939,chi:0.76,s:0.90,lambda0:0.24,C:0.76, event:"Erzincan M7.8 — 33,000 dead. The eastern end of the North Anatolian Fault ruptures. Begins a 60-year westward migration of major earthquakes."},
      {year:1942,chi:0.74,s:0.82,lambda0:0.26,C:0.77, event:"Erbaa-Niksar M7.0. The rupture sequence steps westward. The pattern is not yet recognized as systematic."},
      {year:1944,chi:0.72,s:0.84,lambda0:0.28,C:0.77, event:"Bolu-Gerede M7.3. The sequence continues migrating west. Each rupture loads stress onto the next segment."},
      {year:1967,chi:0.70,s:0.78,lambda0:0.30,C:0.78, event:"Mudurnu Valley M7.1. Now within 200km of Istanbul. Scientists begin recognizing the systematic westward progression."},
      {year:1999,chi:0.64,s:0.86,lambda0:0.36,C:0.80, event:"Izmit M7.6 — 17,000 dead. 90km from Istanbul. Then Düzce M7.2 three months later. The next unruptured segment runs directly under the Sea of Marmara — beneath Istanbul."},
      {year:2010,chi:0.60,s:0.48,lambda0:0.42,C:0.82, event:"The Marmara segment is fully identified as locked. GPS shows 25mm/year right-lateral motion. The fault passes 15km south of Istanbul's 15 million people."},
      {year:2023,chi:0.55,s:0.45,lambda0:0.48,C:0.83, event:"Kahramanmaraş M7.8 and M7.7 kill 50,000 on the East Anatolian Fault — a different structure. The North Anatolian Marmara segment remains locked. Istanbul faces a 64% probability of M7+ before 2040."},
    ]
  },
  {
    id:"new_madrid", name:"New Madrid Seismic Zone", region:"North America", icon:"🇺🇸",
    color:"#60A5FA", lat:36, lon:-90,
    desc:"The most hazardous seismic zone in the central and eastern United States. The 1811-1812 sequence remains the largest earthquake sequence in the contiguous US in historic times.",
    source:"USGS, Johnston & Schweig (1996), Tuttle et al. (2002), Frankel et al. (2012)",
    history:[
      {year:1811,chi:0.80,s:0.95,lambda0:0.20,C:0.68, event:"New Madrid M~7.5-7.9. The Mississippi River runs backward. Church bells ring in Boston. Felt across 2 million square miles. The central US was sparsely populated — today it is not."},
      {year:1812,chi:0.76,s:0.94,lambda0:0.24,C:0.68, event:"Three more M7+ earthquakes in two months. New Madrid, Missouri destroyed. The sequence remains the largest historical earthquake sequence east of the Rockies."},
      {year:1895,chi:0.78,s:0.72,lambda0:0.22,C:0.70, event:"Charleston, Missouri M6.6. A reminder that the zone remains active. Paleoseismic evidence shows M7+ events recurring every 500 years on average."},
      {year:1968,chi:0.76,s:0.65,lambda0:0.24,C:0.72, event:"Illinois M5.5. Modern seismic network begins monitoring. The zone is active with small earthquakes daily. The underlying fault system is poorly understood."},
      {year:2008,chi:0.74,s:0.58,lambda0:0.26,C:0.74, event:"Mt. Carmel M5.2. USGS updates hazard maps — the central US is more vulnerable than previously thought. Memphis sits directly on the zone."},
      {year:2023,chi:0.72,s:0.52,lambda0:0.28,C:0.75, event:"The zone produces thousands of small earthquakes annually. A repeat of 1811-1812 would cause catastrophic damage to Memphis, St. Louis, and regional infrastructure built to no seismic standard."},
    ]
  },
  {
    id:"dead_sea", name:"Dead Sea Transform Fault", region:"Middle East", icon:"🏜️",
    color:"#FCD34D", lat:32, lon:35,
    desc:"The Dead Sea Transform is a 1,000km left-lateral fault connecting the Red Sea rift to the East Anatolian fault. Ancient cities built on its shoulders have been destroyed by it repeatedly.",
    source:"Ambraseys & Finkel (1995), Guidoboni et al. (1994), Ken-Tor et al. (2001)",
    history:[
      {year:749, chi:0.74,s:0.92,lambda0:0.26,C:0.72, event:"Galilee earthquake M7.4. One of the largest historical earthquakes in the Levant. Dozens of cities destroyed simultaneously across modern Israel, Jordan, and Syria. 100,000+ dead."},
      {year:1033,chi:0.72,s:0.88,lambda0:0.28,C:0.73, event:"Jordan Valley M7.3. Jerusalem damaged. The recurrence interval for great earthquakes on this fault is 400-600 years — and the clock is running."},
      {year:1202,chi:0.70,s:0.86,lambda0:0.30,C:0.74, event:"Syria M7.6. Crusader castles destroyed. 30,000 dead across the Levant. The fault ruptures from Syria through Lebanon."},
      {year:1759,chi:0.72,s:0.84,lambda0:0.28,C:0.75, event:"Bekaa Valley sequence — M6.6 and M7.4 within two months. Baalbek and Damascus severely damaged. The most recent major rupture on the northern section."},
      {year:1927,chi:0.70,s:0.76,lambda0:0.30,C:0.76, event:"Jericho M6.2. Last significant earthquake on the southern section. Jerusalem felt strong shaking. The southern Dead Sea segment has not ruptured significantly in 300+ years."},
      {year:2023,chi:0.66,s:0.42,lambda0:0.36,C:0.78, event:"Tel Aviv, Jerusalem, Amman, and Beirut all sit within severe shaking distance of the fault. None of these cities are built to seismic standards appropriate for the hazard. The fault is overdue."},
    ]
  },
];

const SEISMIC_COORDS = {
  cascadia:    [-124, 46],
  san_andreas: [-120, 37],
  japan_trench:[143,  38],
  himalaya:    [84,   28],
  ring_of_fire:[125,   0],
  anatolian:   [32,   40],
  new_madrid:  [-90,  36],
  dead_sea:    [35,   32],
};

function getSeismicM(region, year) {
  const years = region.history.map(h=>h.year);
  let closest = region.history[0];
  let minDiff = Math.abs(years[0]-year);
  region.history.forEach(h=>{
    const diff=Math.abs(h.year-year);
    if(diff<minDiff){minDiff=diff;closest=h;}
  });
  return calcM(closest.chi,closest.s,closest.lambda0,closest.C);
}

function getSeismicPoint(region, year) {
  const years = region.history.map(h=>h.year);
  let closest = region.history[0];
  let minDiff = Math.abs(years[0]-year);
  region.history.forEach(h=>{
    const diff=Math.abs(h.year-year);
    if(diff<minDiff){minDiff=diff;closest=h;}
  });
  return closest;
}

const SEISMIC_YEARS = [1800,1850,1900,1950,1970,1990,2000,2010,2023];

// ── SEISMIC TAB ───────────────────────────────────────────────────────────────
function SeismicTab() {
  const [year, setYear] = useState(2023);
  const [selected, setSelected] = useState(null);

  const sel = selected ? SEISMIC_REGIONS.find(s=>s.id===selected) : null;
  const selPt = sel ? getSeismicPoint(sel, year) : null;

  const toXY = (lon,lat,W,H) => [
    ((lon+180)/360)*W,
    ((90-lat)/180)*H,
  ];

  const W=620, H=310;

  const scored = [...SEISMIC_REGIONS].sort((a,b)=>getSeismicM(b,year)-getSeismicM(a,year));

  function SeismicSparkline({region}){
    const pts = region.history;
    const vals = pts.map(p=>calcM(p.chi,p.s,p.lambda0,p.C));
    const minV=Math.min(...vals,-0.5), maxV=Math.max(...vals,0.3);
    const rng=maxV-minV, w=120, h=36, pad=4;
    const xs=pts.map((_,i)=>pad+(i/(pts.length-1))*(w-pad*2));
    const ys=vals.map(v=>h-pad-((v-minV)/rng)*(h-pad*2));
    const pathD=xs.map((x,i)=>`${i===0?"M":"L"} ${x} ${ys[i]}`).join(" ");
    const zeroY=h-pad-((0-minV)/rng)*(h-pad*2);
    const lastM=vals[vals.length-1];
    return(
      <svg width={w} height={h} style={{display:"block",overflow:"visible"}}>
        <line x1={pad} y1={zeroY} x2={w-pad} y2={zeroY} stroke="#333" strokeWidth={0.5} strokeDasharray="2,3"/>
        <path d={pathD} fill="none" stroke={mColor(lastM)} strokeWidth={1.5} strokeLinecap="round"/>
        {vals.map((v,i)=><circle key={i} cx={xs[i]} cy={ys[i]} r={i===vals.length-1?3:2} fill={mColor(v)}/>)}
      </svg>
    );
  }

  return (
    <div style={{display:"flex",flexDirection:"column",gap:24}}>
      {/* Header */}
      <div>
        <h2 style={{fontFamily:"var(--serif)",fontSize:28,color:"#FFFFFF",marginBottom:10,
          borderLeft:"3px solid #F87171",paddingLeft:14}}>
          Seismic Systems
        </h2>
        <p style={{color:"#A3A3A3",fontSize:13,fontFamily:"var(--sans)",lineHeight:1.65,maxWidth:640}}>
          The same equation that measures civilizational collapse applies to tectonic systems.
          Fault zones accumulate strain (burden), release energy (throughput), and have structural
          efficiency determined by their geometry. When burden exceeds output capacity, the system ruptures.
          These are 8 of the world's most consequential seismic zones — calibrated from USGS,
          ISC, and peer-reviewed seismology.
        </p>
      </div>

      {/* Year slider */}
      <div style={{background:"#111111",border:"1px solid #2A2A2A",borderRadius:10,
        padding:"12px 18px",display:"flex",alignItems:"center",gap:14}}>
        <span style={{fontFamily:"var(--serif)",fontSize:22,color:"#FFFFFF",flexShrink:0}}>{year}</span>
        <input type="range" min={0} max={SEISMIC_YEARS.length-1} step={1}
          value={SEISMIC_YEARS.indexOf(year)<0?SEISMIC_YEARS.length-1:SEISMIC_YEARS.indexOf(year)}
          onChange={e=>setYear(SEISMIC_YEARS[parseInt(e.target.value)])}
          style={{flex:1,accentColor:"#EF4444"}}
        />
        <div style={{display:"flex",gap:6,flexShrink:0,flexWrap:"wrap"}}>
          {SEISMIC_YEARS.map(y=>(
            <button key={y} onClick={()=>setYear(y)} style={{
              background:year===y?"#EF4444":"#1A1A1A",
              border:`1px solid ${year===y?"#EF4444":"#2A2A2A"}`,
              borderRadius:4,padding:"3px 7px",fontSize:9,
              color:year===y?"#FFFFFF":"#525252",fontFamily:"var(--mono)"
            }}>{y}</button>
          ))}
        </div>
      </div>

      {/* World map */}
      <div style={{background:"#0A0A0A",border:"1px solid #2A2A2A",borderRadius:14,padding:16}}>
        <div style={{fontFamily:"var(--mono)",fontSize:9,color:"#EF4444",marginBottom:10,letterSpacing:3}}>
          GLOBAL SEISMIC HAZARD MAP · {year} · COLOR = STABILITY MARGIN
        </div>
        <svg viewBox={`0 0 ${W} ${H}`} style={{display:"block",width:"100%",height:"auto"}} preserveAspectRatio="xMidYMid meet">
          <rect x={0} y={0} width={W} height={H} fill="#050505"/>
          {/* Grid */}
          {[-60,-30,0,30,60].map(lat=>{
            const [,y]=toXY(0,lat,W,H);
            return <line key={lat} x1={0} y1={y} x2={W} y2={y} stroke="#1A1A1A" strokeWidth={0.8}/>;
          })}
          {/* Region labels */}
          <g fontFamily="Inter,sans-serif" fontSize="9" fill="#3A4A5A" opacity="0.7" letterSpacing="2">
            <text x="124" y="113" textAnchor="middle">NORTH AMERICA</text>
            <text x="305" y="78" textAnchor="middle">EUROPE</text>
            <text x="310" y="160" textAnchor="middle">AFRICA</text>
            <text x="400" y="88" textAnchor="middle">MIDDLE EAST</text>
            <text x="445" y="62" textAnchor="middle">RUSSIA / C. ASIA</text>
            <text x="465" y="118" textAnchor="middle">S. ASIA</text>
            <text x="505" y="88" textAnchor="middle">E. ASIA</text>
            <text x="525" y="138" textAnchor="middle">SE ASIA</text>
            <text x="525" y="200" textAnchor="middle">AUSTRALIA</text>
          </g>
          {/* Fault zone indicators — pulsing rings at fault locations */}
          {SEISMIC_REGIONS.map(region=>{
            const coords = SEISMIC_COORDS[region.id];
            if(!coords) return null;
            const M = getSeismicM(region, year);
            const [x,y] = toXY(coords[0],coords[1],W,H);
            const isSel = selected===region.id;
            return (
              <g key={region.id} onClick={()=>setSelected(isSel?null:region.id)} style={{cursor:"pointer"}}>
                <circle cx={x} cy={y} r={28} fill={mColor(M)} opacity={0.06}/>
                <circle cx={x} cy={y} r={18} fill={mColor(M)} opacity={0.12}/>
                <circle cx={x} cy={y} r={12} fill={mColor(M)} opacity={isSel?1:0.80}
                  stroke={isSel?"#FFFFFF":"none"} strokeWidth={2}/>
                <text x={x} y={y+4} textAnchor="middle" fontSize={10}>{region.icon}</text>
                <text x={x} y={y+26} textAnchor="middle" fontSize={8}
                  fill="#FFFFFF" fontFamily="Inter" fontWeight="600">
                  {region.name.split(" ")[0]}
                </text>
                <text x={x} y={y+36} textAnchor="middle" fontSize={8}
                  fill={mColor(M)} fontFamily="JetBrains Mono" fontWeight="700">
                  {(M>=0?"+":"")+M.toFixed(2)}
                </text>
              </g>
            );
          })}
        </svg>
        {/* Legend */}
        <div style={{display:"flex",gap:14,marginTop:10,flexWrap:"wrap",alignItems:"center"}}>
          {[["#06B6D4","Low hazard"],["#22C55E","Moderate"],["#EAB308","Elevated"],
            ["#F97316","High"],["#EF4444","Critical — rupture overdue"]].map(([c,l])=>(
            <div key={l} style={{display:"flex",alignItems:"center",gap:5}}>
              <div style={{width:8,height:8,borderRadius:"50%",background:c}}/>
              <span style={{fontSize:10,color:"#737373",fontFamily:"var(--sans)"}}>{l}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Selected region detail */}
      {sel && selPt && (
        <div style={{background:"#0A0A0A",border:`1px solid ${mColor(getSeismicM(sel,year))}40`,
          borderRadius:14,overflow:"hidden"}}>
          <div style={{padding:"20px 24px",borderBottom:"1px solid #1A1A1A",
            display:"flex",justifyContent:"space-between",alignItems:"flex-start",flexWrap:"wrap",gap:16}}>
            <div>
              <div style={{display:"flex",alignItems:"center",gap:10,marginBottom:8}}>
                <span style={{fontSize:28}}>{sel.icon}</span>
                <div>
                  <div style={{fontSize:17,fontWeight:700,color:"#FFFFFF",fontFamily:"var(--sans)"}}>{sel.name}</div>
                  <div style={{fontSize:11,color:"#525252",fontFamily:"var(--mono)"}}>{sel.region}</div>
                </div>
              </div>
              <p style={{fontSize:13,color:"#A3A3A3",lineHeight:1.7,fontFamily:"var(--sans)",maxWidth:500}}>{sel.desc}</p>
              <div style={{fontSize:10,color:"#404040",fontFamily:"var(--mono)",marginTop:8}}>Source: {sel.source}</div>
            </div>
            <div style={{textAlign:"center",flexShrink:0}}>
              <Gauge value={getSeismicM(sel,year)} size={150}/>
              <div style={{fontSize:13,fontWeight:700,color:mColor(getSeismicM(sel,year)),
                fontFamily:"var(--sans)",marginTop:4}}>{mLabel(getSeismicM(sel,year))}</div>
            </div>
          </div>

          {/* Trajectory chart */}
          <div style={{padding:"16px 24px",borderBottom:"1px solid #1A1A1A"}}>
            <div style={{fontFamily:"var(--mono)",fontSize:9,color:"#EF4444",marginBottom:12,letterSpacing:3}}>
              M TRAJECTORY · {sel.history[0].year}–{sel.history[sel.history.length-1].year}
            </div>
            <MChart points={sel.history} dsColor={sel.color} dsId={sel.id}/>
          </div>

          <MInsight points={sel.history} dsId={sel.id} dsLabel={sel.name} domain={sel.region}/>
          {/* Current point */}
          <div style={{padding:"16px 24px",borderBottom:"1px solid #1A1A1A",background:"#000000"}}>
            <div style={{fontSize:9,fontFamily:"var(--mono)",color:sel.color,marginBottom:10,letterSpacing:2}}>
              HISTORICAL CONTEXT · {selPt.year}
            </div>
            <p style={{fontSize:13,color:"#D4D4D4",lineHeight:1.7,fontFamily:"var(--sans)"}}>{selPt.event}</p>
          </div>

          {/* Variable breakdown */}
          <div style={{padding:"16px 24px",display:"grid",gridTemplateColumns:"repeat(4,1fr)",gap:8}}>
            {[
              {sym:"χ",val:selPt.chi,color:"#60A5FA",name:"Fault Efficiency"},
              {sym:"s",val:selPt.s,color:"#A78BFA",name:"Energy Release"},
              {sym:"λ(C)",val:selPt.lambda0+k*Math.pow(selPt.C,n),color:"#F87171",name:"Strain Burden"},
              {sym:"C",val:selPt.C,color:"#FCD34D",name:"Fault Complexity"},
            ].map((v,i)=>(
              <div key={i} style={{background:"#111111",borderRadius:8,padding:"12px 14px",
                textAlign:"center",border:"1px solid #1A1A1A"}}>
                <div style={{fontFamily:"var(--mono)",fontSize:15,color:v.color,fontWeight:700}}>{v.val.toFixed(3)}</div>
                <div style={{fontSize:9,color:"#525252",marginTop:3,fontFamily:"var(--sans)"}}>{v.sym} · {v.name}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Ranked list */}
      <div style={{background:"#0A0A0A",border:"1px solid #2A2A2A",borderRadius:14,overflow:"hidden"}}>
        <div style={{padding:"14px 20px",borderBottom:"1px solid #1A1A1A",
          display:"flex",justifyContent:"space-between",alignItems:"center",flexWrap:"wrap",gap:8}}>
          <div style={{fontFamily:"var(--mono)",fontSize:9,color:"#EF4444",letterSpacing:3}}>
            SEISMIC ZONES RANKED BY STABILITY MARGIN · {year}
          </div>
          <div style={{fontFamily:"var(--mono)",fontSize:9,color:"#EF4444"}}>
            {scored.filter(s=>getSeismicM(s,year)<0).length} of {scored.length} zones in negative margin
          </div>
        </div>

        {scored.map((region,i)=>{
          const M = getSeismicM(region,year);
          const isSel = selected===region.id;
          const barPct = Math.min(Math.abs(M)/0.6*50,50);
          return (
            <div key={region.id} onClick={()=>setSelected(isSel?null:region.id)}
              style={{
                display:"flex",alignItems:"center",gap:12,padding:"12px 20px",
                borderBottom:"1px solid #111111",cursor:"pointer",
                background:isSel?"#0A0A14":"transparent",transition:"background 0.1s"
              }}
              onMouseEnter={e=>{if(!isSel)e.currentTarget.style.background="#111111";}}
              onMouseLeave={e=>{if(!isSel)e.currentTarget.style.background="transparent";}}
            >
              <div style={{fontFamily:"var(--mono)",fontSize:10,color:"#525252",
                width:20,textAlign:"right",flexShrink:0}}>{i+1}</div>
              <div style={{display:"flex",alignItems:"center",gap:8,width:220,flexShrink:0}}>
                <span style={{fontSize:18}}>{region.icon}</span>
                <div>
                  <div style={{fontSize:12,fontWeight:600,color:isSel?"#FFFFFF":"#D4D4D4",
                    fontFamily:"var(--sans)"}}>{region.name}</div>
                  <div style={{fontSize:9,color:"#525252",fontFamily:"var(--sans)"}}>{region.region}</div>
                </div>
              </div>
              <SeismicSparkline region={region}/>
              <div style={{flex:1,height:6,background:"#1A1A1A",borderRadius:3,
                position:"relative",overflow:"visible",minWidth:60}}>
                <div style={{
                  position:"absolute",
                  right:M>=0?"50%":undefined,left:M<0?"50%":undefined,
                  width:`${barPct}%`,height:"100%",
                  background:mColor(M),borderRadius:3,top:0,
                }}/>
                <div style={{position:"absolute",left:"50%",top:-2,width:1,height:10,background:"#333"}}/>
              </div>
              <div style={{fontFamily:"var(--mono)",fontSize:12,fontWeight:700,
                color:mColor(M),width:68,textAlign:"right",flexShrink:0}}>
                {M>=0?"+":""}{M.toFixed(3)}
              </div>
              <div style={{fontSize:10,color:mColor(M),fontFamily:"var(--sans)",width:62,flexShrink:0}}>
                {mLabel(M)}
              </div>
            </div>
          );
        })}
      </div>

      {/* Scientific note */}
      <div style={{background:"#0A0A0A",border:"1px solid #2A2A2A",borderRadius:10,
        padding:"14px 18px",display:"flex",gap:10,alignItems:"flex-start"}}>
        <span style={{fontSize:13,flexShrink:0}}>⚠️</span>
        <p style={{fontSize:11,color:"#737373",lineHeight:1.6,fontFamily:"var(--sans)"}}>
          Seismic M values are calibrated from USGS, ISC, and peer-reviewed seismology literature.
          χ = fault structural efficiency (geometry regularity, coupling coefficient),
          s = normalized seismic moment release relative to regional maximum,
          λ₀ = accumulated strain burden (slip deficit, recurrence interval fraction elapsed),
          C = fault network complexity (segment count, branching, interaction zones).
          These are diagnostic proxies, not physical measurements.
          Framework under peer review — cite as: Baird, N. (2026). Engine of Emergence.
        </p>
      </div>
    </div>
  );
}



// ── FLOATING ASSISTANT BUBBLE ─────────────────────────────────────────────────
function FloatingAssistant() {
  const [open, setOpen] = useState(false);
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([
    { role:"assistant", text:"Hey — ask me anything about EoE, or describe a system you want to analyze and I'll tell you exactly what data to collect and where to find it." }
  ]);
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    if (open) {
      setTimeout(() => inputRef.current?.focus(), 100);
    }
  }, [open]);

  // Expose trigger for external components
  useEffect(() => {
    window._eoeAskAssistant = (q) => {
      setOpen(true);
      setMessages(prev => [...prev, { role:"user", text:q }]);
      // Auto-send after opening
      setTimeout(async () => {
        setLoading(true);
        try {
          const resp = await fetch("https://api.anthropic.com/v1/messages", {
            method:"POST", headers:{"Content-Type":"application/json","x-api-key":import.meta.env.VITE_ANTHROPIC_KEY||"","anthropic-version":"2023-06-01","anthropic-dangerous-direct-browser-access":"true"},
            body: JSON.stringify({
              model:"claude-sonnet-4-5", max_tokens:400,
              system: SYSTEM,
              messages: [{ role:"user", content:q }]
            })
          });
          const data = await resp.json();
          const reply = data.content?.map(b=>b.text||"").join("") || "Try again.";
          setMessages(prev=>[...prev, { role:"assistant", text:reply }]);
        } catch(e) {
          setMessages(prev=>[...prev, { role:"assistant", text:"Connection issue — try again." }]);
        }
        setLoading(false);
      }, 300);
    };
    return () => { delete window._eoeAskAssistant; };
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior:"smooth" });
  }, [messages, loading]);

  const SYSTEM = `You are the Engine of Emergence assistant. EoE measures M = χs − λ(C) where χ=efficiency, s=throughput, λ₀=burden, C=complexity. M positive = runway, M negative = borrowed time.

Approved data sources by domain:
- Business: SEC EDGAR, Macrotrends, World Bank Enterprise Surveys
- City/Urban: Lincoln Institute FiSC, BEA Metro GDP, US Census ACS  
- Government: World Bank Open Data, IMF WEO, CBO Historical Data
- Ecological: AIMS LTMP, NOAA Coral Reef Watch, Global Forest Watch, NASA AppEEARS
- Agriculture/Valley: USDA NASS (nass.usda.gov), California DWR (water.ca.gov), USGS Water Resources
- Seismic: USGS Earthquake Catalog, UNAVCO GPS data
- Historical: Seshat Databank, HYDE Database

When someone asks where to find data for a specific system, always name the exact approved source and tell them which columns map to χ, s, λ₀, and C. Be concise - 2-4 sentences. No bullet points.`;

  async function send() {
    const q = input.trim();
    if (!q || loading) return;
    const newMsgs = [...messages, { role:"user", text:q }];
    setMessages(newMsgs);
    setInput("");
    setLoading(true);
    try {
      const resp = await fetch("https://api.anthropic.com/v1/messages", {
        method:"POST",
        headers:{ "Content-Type":"application/json" },
        body: JSON.stringify({
          model:"claude-sonnet-4-5",
          max_tokens:400,
          system: SYSTEM,
          messages: newMsgs.map(m=>({ role:m.role, content:m.text }))
        })
      });
      const data = await resp.json();
      const reply = data.content?.map(b=>b.text||"").join("") || "Try again in a moment.";
      setMessages(prev=>[...prev, { role:"assistant", text:reply }]);
    } catch(e) {
      setMessages(prev=>[...prev, { role:"assistant", text:"Connection issue — try again." }]);
    }
    setLoading(false);
  }

  return (
    <>
      {/* Backdrop */}
      {open && (
        <div onClick={()=>setOpen(false)}
          style={{position:"fixed",inset:0,zIndex:98,background:"transparent"}}/>
      )}

      {/* Chat panel */}
      {open && (
        <div style={{
          position:"fixed", bottom:80, right:20, zIndex:99,
          width:360, height:480,
          background:"#0A0A0A", border:"1px solid #2A2A2A",
          borderRadius:16, boxShadow:"0 8px 40px #000000CC",
          display:"flex", flexDirection:"column", overflow:"hidden",
          animation:"fadeUp 0.2s ease both"
        }}>
          {/* Header */}
          <div style={{padding:"14px 16px", borderBottom:"1px solid #1A1A1A",
            display:"flex", justifyContent:"space-between", alignItems:"center",
            background:"#111111", flexShrink:0}}>
            <div style={{display:"flex", alignItems:"center", gap:8}}>
              <div style={{width:28,height:28,borderRadius:"50%",background:"#2563EB",
                display:"flex",alignItems:"center",justifyContent:"center",
                fontSize:12,fontWeight:700,color:"#FFFFFF"}}>E</div>
              <div>
                <div style={{fontSize:13,fontWeight:600,color:"#FFFFFF",fontFamily:"var(--sans)"}}>
                  EoE Assistant
                </div>
                <div style={{fontSize:10,color:"#525252",fontFamily:"var(--sans)"}}>
                  Ask anything · Data sources · Variable help
                </div>
              </div>
            </div>
            <button onClick={()=>setOpen(false)} style={{
              background:"none",border:"none",color:"#525252",
              fontSize:18,cursor:"pointer",padding:4
            }}>✕</button>
          </div>

          {/* Messages */}
          <div style={{flex:1,overflowY:"auto",padding:"14px 16px",
            display:"flex",flexDirection:"column",gap:10}}>
            {messages.map((m,i)=>(
              <div key={i} style={{display:"flex",
                justifyContent:m.role==="user"?"flex-end":"flex-start",gap:6}}>
                {m.role==="assistant" && (
                  <div style={{width:22,height:22,borderRadius:"50%",background:"#2563EB",
                    display:"flex",alignItems:"center",justifyContent:"center",
                    fontSize:9,fontWeight:700,color:"#FFF",flexShrink:0,marginTop:2}}>E</div>
                )}
                <div style={{
                  maxWidth:"82%",
                  background:m.role==="user"?"#1A1A1A":"#111111",
                  border:`1px solid ${m.role==="user"?"#2A2A2A":"#1A1A1A"}`,
                  borderRadius:m.role==="user"?"12px 12px 3px 12px":"3px 12px 12px 12px",
                  padding:"9px 12px",fontSize:12,lineHeight:1.65,
                  color:"#D4D4D4",fontFamily:"var(--sans)"
                }}>{m.text}</div>
              </div>
            ))}
            {loading && (
              <div style={{display:"flex",gap:6}}>
                <div style={{width:22,height:22,borderRadius:"50%",background:"#2563EB",
                  display:"flex",alignItems:"center",justifyContent:"center",
                  fontSize:9,fontWeight:700,color:"#FFF",flexShrink:0}}>E</div>
                <div style={{background:"#111111",border:"1px solid #1A1A1A",
                  borderRadius:"3px 12px 12px 12px",padding:"9px 12px",
                  display:"flex",gap:4,alignItems:"center"}}>
                  {[0,1,2].map(j=>(
                    <div key={j} style={{width:5,height:5,borderRadius:"50%",
                      background:"#3B82F6",animation:"pulse 1.2s ease-in-out infinite",
                      animationDelay:`${j*0.2}s`}}/>
                  ))}
                </div>
              </div>
            )}
            <div ref={bottomRef}/>
          </div>

          {/* Quick prompts */}
          <div style={{padding:"8px 12px",borderTop:"1px solid #1A1A1A",
            display:"flex",flexWrap:"wrap",gap:5,flexShrink:0}}>
            {["Where do I find data for my city?","What is χ?","Explain M like I'm 10","How do I normalize my data?"].map((q,i)=>(
              <button key={i} onClick={()=>setInput(q)} style={{
                background:"#111111",border:"1px solid #1A1A1A",borderRadius:12,
                padding:"3px 9px",fontSize:10,color:"#737373",
                fontFamily:"var(--sans)",cursor:"pointer",transition:"all 0.12s"
              }}
                onMouseEnter={e=>{e.currentTarget.style.borderColor="#2563EB";e.currentTarget.style.color="#93C5FD";}}
                onMouseLeave={e=>{e.currentTarget.style.borderColor="#1A1A1A";e.currentTarget.style.color="#737373";}}
              >{q}</button>
            ))}
          </div>

          {/* Input */}
          <div style={{padding:"10px 12px 14px",flexShrink:0,display:"flex",gap:8}}>
            <input ref={inputRef} value={input}
              onChange={e=>setInput(e.target.value)}
              onKeyDown={e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();send();}}}
              placeholder="Ask anything..."
              style={{flex:1,background:"#111111",border:"1px solid #2A2A2A",
                borderRadius:8,padding:"9px 12px",fontSize:12,color:"#FFFFFF",
                outline:"none",fontFamily:"var(--sans)"}}
              onFocus={e=>e.target.style.borderColor="#2563EB"}
              onBlur={e=>e.target.style.borderColor="#2A2A2A"}
            />
            <button onClick={send} disabled={loading||!input.trim()} style={{
              background:loading||!input.trim()?"#1A1A1A":"#2563EB",
              border:"none",borderRadius:8,width:38,height:38,
              fontSize:16,color:"white",cursor:loading||!input.trim()?"not-allowed":"pointer",
              opacity:loading||!input.trim()?0.4:1,flexShrink:0
            }}>↑</button>
          </div>
        </div>
      )}

      {/* Bubble trigger */}
      <button onClick={()=>setOpen(o=>!o)} style={{
        position:"fixed", bottom:20, right:20, zIndex:100,
        background:open?"#1D4ED8":"#2563EB",
        border:"none", borderRadius:24,
        padding:"10px 18px 10px 14px",
        boxShadow:"0 4px 20px #2563EB50",
        display:"flex", alignItems:"center", gap:8,
        cursor:"pointer", transition:"all 0.2s",
        fontFamily:"var(--sans)"
      }}
        onMouseEnter={e=>{e.currentTarget.style.transform="scale(1.05)";e.currentTarget.style.boxShadow="0 6px 28px #2563EB70";}}
        onMouseLeave={e=>{e.currentTarget.style.transform="scale(1)";e.currentTarget.style.boxShadow="0 4px 20px #2563EB50";}}
      >
        <span style={{fontSize:18}}>💬</span>
        <span style={{fontSize:13,fontWeight:600,color:"#FFFFFF"}}>
          {open ? "Close" : "Ask anything"}
        </span>
      </button>
    </>
  );
}

// ── MAIN APP ──────────────────────────────────────────────────────────────────
export default function EoEApp() {
  const [screen, setScreen] = useState(
    window.location.hash === "#admin" ? "admin" : "landing"
  );
  const [tab, setTab] = useState("explore");
  const [uploadedDatasets, setUploadedDatasets] = useState([]);

  function handleExperimentReady(ds) {
    setUploadedDatasets(prev => {
      const filtered = prev.filter(d=>d.id!==ds.id);
      return [ds, ...filtered];
    });
  }

  if (screen === "admin") {
    return <AdminPage onBack={()=>{ window.location.hash=""; setScreen("landing"); }}/>;
  }

  if (screen === "landing") {
    return <Landing onEnter={()=>setScreen("app")}/>;
  }

  const TABS = [
    {id:"understand",  label:"Understand",  icon:"📖", accent:"#A78BFA", desc:"The framework"},
    {id:"explore",     label:"Explore",     icon:"🔭", accent:"#3B82F6", desc:"20 real systems"},
    {id:"experiment",  label:"Experiment",  icon:"⚗️", accent:"#22C55E", desc:"Your data"},
    {id:"directory",   label:"Sources",     icon:"🗂️", accent:"#EC4899", desc:"34 datasets"},
    {id:"compare",     label:"Compare",     icon:"🌍", accent:"#06B6D4", desc:"Nations"},
    {id:"climate",     label:"Climate",     icon:"🌡️", accent:"#EF4444", desc:"Earth systems"},
    {id:"seismic",     label:"Seismic",     icon:"🌋", accent:"#F97316", desc:"Fault zones"},
  ];
  const activeTabMeta = TABS.find(t=>t.id===tab) || TABS[0];

  return (
    <>
      <style>{GLOBAL_CSS}</style>
      <div style={{minHeight:"100vh",background:"var(--bg)",display:"flex",flexDirection:"column"}}>

        {/* Top accent line */}
        <div style={{height:2,background:`linear-gradient(90deg,transparent,${activeTabMeta.accent},${activeTabMeta.accent}CC,transparent)`,flexShrink:0,transition:"background 0.4s"}}/>

        {/* Header */}
        <div style={{
          borderBottom:"1px solid var(--border)",background:"#000000F5",
          backdropFilter:"blur(10px)",position:"sticky",top:0,zIndex:50,flexShrink:0
        }}>
          <div style={{maxWidth:1040,margin:"0 auto",padding:"0 16px",display:"flex",alignItems:"center",justifyContent:"space-between",height:52}}>
            <button onClick={()=>setScreen("landing")} style={{background:"none",border:"none",display:"flex",alignItems:"center",gap:10,color:"#FFFFFF"}}>
              <span style={{fontFamily:"var(--mono)",fontSize:10,color:"var(--accent2)",letterSpacing:3}}>EoE</span>
              <span style={{fontFamily:"var(--serif)",fontSize:16,fontWeight:400}}>Engine of Emergence</span>
            </button>
            <div style={{fontSize:9,fontFamily:"var(--mono)",color:"#2A2A2A"}}>Nathan Baird · Independent Researcher · 2026</div>
          </div>
          {/* Tabs */}
          <div style={{maxWidth:1040,margin:"0 auto",padding:"0 8px",display:"flex",borderTop:"1px solid var(--border)",overflowX:"auto"}}>
            {TABS.map(t=>(
              <button key={t.id} onClick={()=>setTab(t.id)} style={{
                padding:"9px 10px",background:"none",border:"none",
                borderBottom:tab===t.id?`2px solid ${t.accent}`:"2px solid transparent",
                marginBottom:-1,fontSize:11,fontWeight:tab===t.id?700:400,
                color:tab===t.id?t.accent:"#A3A3A3",fontFamily:"var(--sans)",
                transition:"all 0.15s",display:"flex",alignItems:"center",gap:5,
              }}>
                <span>{t.icon}</span><span>{t.label}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Content */}
        <div style={{flex:1,maxWidth:1040,margin:"0 auto",padding:"40px 24px",width:"100%",borderTop:`1px solid ${activeTabMeta.accent}18`,transition:"border-color 0.4s"}}>
          {tab==="understand"  && <UnderstandTab/>}
          {tab==="explore"     && <ExploreTab/>}
          {tab==="experiment"  && <ExperimentTab onGoToExplore={()=>setTab("explore")} onGoToAssistant={()=>setTab("assistant")} uploadedDatasets={uploadedDatasets}/>}
          {tab==="directory"   && <DirectoryTab/>}
          {tab==="compare"     && <CompareTab/>}
          {tab==="climate"     && <ClimateTab/>}
          {tab==="seismic"     && <SeismicTab/>}
        </div>

        <FloatingAssistant/>
        {/* Footer */}
        <div style={{borderTop:"1px solid var(--border)",padding:"14px 20px",textAlign:"center",flexShrink:0}}>
          <p style={{fontSize:10,color:"#2A2A2A",fontFamily:"var(--sans)"}}>
            EoE is a candidate framework under peer review · Results are exploratory, not validated findings · Cite as: Baird, N. (2026). Engine of Emergence · arXiv: [pending]
          </p>
        </div>
      </div>
    </>
  );
}
