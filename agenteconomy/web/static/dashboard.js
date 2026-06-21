const state = {
  runs: [],
  selectedRun: null,
  records: [],
  events: [],
  stage: {},
  replayIndex: 0,
  playTimer: null,
};

const el = {
  runSelect: document.getElementById("runSelect"),
  refreshBtn: document.getElementById("refreshBtn"),
  runPath: document.getElementById("runPath"),
  liveDot: document.getElementById("liveDot"),
  activeStage: document.getElementById("activeStage"),
  activeMonth: document.getElementById("activeMonth"),
  updatedAt: document.getElementById("updatedAt"),
  accountingState: document.getElementById("accountingState"),
  replayTitle: document.getElementById("replayTitle"),
  prevBtn: document.getElementById("prevBtn"),
  playBtn: document.getElementById("playBtn"),
  nextBtn: document.getElementById("nextBtn"),
  speedSelect: document.getElementById("speedSelect"),
  monthRange: document.getElementById("monthRange"),
  metricGrid: document.getElementById("metricGrid"),
  stageTimeline: document.getElementById("stageTimeline"),
  stageRows: document.getElementById("stageRows"),
  eventCount: document.getElementById("eventCount"),
  seriesChart: document.getElementById("seriesChart"),
  profitGrid: document.getElementById("profitGrid"),
  profitState: document.getElementById("profitState"),
  gapList: document.getElementById("gapList"),
  creditGrid: document.getElementById("creditGrid"),
  accountingGrid: document.getElementById("accountingGrid"),
};

function fmtNumber(value, digits = 0) {
  const num = Number(value);
  if (!Number.isFinite(num)) return "--";
  return new Intl.NumberFormat("zh-CN", {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  }).format(num);
}

function fmtMoney(value) {
  const num = Number(value);
  if (!Number.isFinite(num)) return "--";
  const abs = Math.abs(num);
  if (abs >= 1_000_000) return `$${fmtNumber(num / 1_000_000, 2)}M`;
  if (abs >= 1_000) return `$${fmtNumber(num / 1_000, 1)}K`;
  return `$${fmtNumber(num, 0)}`;
}

function fmtPct(value) {
  const num = Number(value);
  if (!Number.isFinite(num)) return "--";
  return `${fmtNumber(num * 100, 1)}%`;
}

function fmtTime(value) {
  if (!value) return "--";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString("zh-CN", { hour12: false });
}

function metric(label, value, tone = "") {
  return `<div class="metric ${tone}"><span>${label}</span><strong>${value}</strong></div>`;
}

function detail(label, value) {
  return `<div class="detail-item"><span>${label}</span><strong>${value}</strong></div>`;
}

async function fetchJson(url) {
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

async function loadRuns(keepSelection = true) {
  const data = await fetchJson("/api/runs?limit=100");
  state.runs = data.runs || [];
  const previous = keepSelection ? state.selectedRun : null;
  state.selectedRun =
    previous && state.runs.some((run) => run.id === previous)
      ? previous
      : state.runs[0]?.id || null;
  renderRunSelect();
  if (state.selectedRun) {
    await loadRun(state.selectedRun);
  } else {
    renderEmpty();
  }
}

function renderRunSelect() {
  el.runSelect.innerHTML = "";
  if (!state.runs.length) {
    const opt = document.createElement("option");
    opt.textContent = "未发现运行记录";
    opt.value = "";
    el.runSelect.appendChild(opt);
    return;
  }
  for (const run of state.runs) {
    const opt = document.createElement("option");
    opt.value = run.id;
    opt.textContent = `${run.id} · ${run.record_count} records`;
    opt.selected = run.id === state.selectedRun;
    el.runSelect.appendChild(opt);
  }
}

async function loadRun(runId) {
  const data = await fetchJson(`/api/run?path=${encodeURIComponent(runId)}`);
  state.selectedRun = runId;
  state.records = data.records || [];
  state.events = data.events || [];
  state.stage = data.stage || {};
  state.replayIndex = Math.min(state.replayIndex, Math.max(0, state.records.length - 1));
  if (!state.records.length) state.replayIndex = 0;
  renderAll();
}

function renderEmpty() {
  el.runPath.textContent = "output 下暂无 run_* 记录";
  el.activeStage.textContent = "无活动阶段";
  el.activeMonth.textContent = "--";
  el.updatedAt.textContent = "--";
  el.accountingState.textContent = "--";
  el.metricGrid.innerHTML = `<p class="muted-empty">等待仿真写入月度记录。</p>`;
  el.stageRows.innerHTML = "";
  el.stageTimeline.innerHTML = "";
  el.eventCount.textContent = "0";
  el.profitGrid.innerHTML = "";
  el.creditGrid.innerHTML = "";
  el.accountingGrid.innerHTML = "";
  clearChart();
}

function renderAll() {
  const record = currentRecord();
  const run = state.runs.find((item) => item.id === state.selectedRun);
  el.runPath.textContent = state.selectedRun || "未选择运行";
  el.updatedAt.textContent = fmtTime(run?.updated_at || record?.mtime || record?.timestamp);
  renderLiveStatus(record);
  renderReplay(record);
  renderStages();
  renderChart();
  renderDetails(record);
}

function currentRecord() {
  if (!state.records.length) return null;
  return state.records[state.replayIndex] || state.records[state.records.length - 1];
}

function stageMonthLabel(event, record) {
  const source = event || record || {};
  const phase = source.preheat ? "预热" : "正式";
  const month = source.month ?? "--";
  const econ = source.econ_month ?? "--";
  return `${phase} M${month} / E${econ}`;
}

function renderLiveStatus(record) {
  const active = state.stage?.active_stage;
  const latest = state.stage?.latest_event;
  const errorCount = Number(state.stage?.error_count || 0);
  el.liveDot.className = "live-dot";
  if (errorCount > 0) el.liveDot.classList.add("error");
  else if (active) el.liveDot.classList.add("live");

  if (active) {
    el.activeStage.textContent = active.stage || "运行中";
    el.activeMonth.textContent = stageMonthLabel(active, record);
  } else if (latest) {
    const status = latest.status === "error" ? "异常" : "最近完成";
    el.activeStage.textContent = `${status}: ${latest.stage || "阶段"}`;
    el.activeMonth.textContent = stageMonthLabel(latest, record);
  } else if (record) {
    el.activeStage.textContent = "月度记录已写入";
    el.activeMonth.textContent = stageMonthLabel(null, record);
  } else {
    el.activeStage.textContent = "无活动阶段";
    el.activeMonth.textContent = "--";
  }

  if (record?.accounting) {
    const ok = record.accounting.ok;
    el.accountingState.textContent = ok ? "OK" : `${record.accounting.error_count || 0} errors`;
  } else {
    el.accountingState.textContent = "--";
  }
}

function renderReplay(record) {
  const max = Math.max(0, state.records.length - 1);
  el.monthRange.max = String(max);
  el.monthRange.value = String(state.replayIndex);
  el.replayTitle.textContent = record
    ? `${record.preheat ? "预热" : "正式"} M${record.month} / E${record.econ_month}`
    : "月度记录";

  if (!record) {
    el.metricGrid.innerHTML = `<p class="muted-empty">暂无月度记录。</p>`;
    return;
  }

  el.metricGrid.innerHTML = [
    metric("名义 GDP", fmtMoney(record.macro?.nominal_gdp)),
    metric("实际 GDP", fmtMoney(record.macro?.real_gdp)),
    metric("价格指数", fmtNumber(record.macro?.price_index, 2)),
    metric("通胀率", fmtPct(record.macro?.inflation_rate)),
    metric("就业率", fmtPct(record.labor?.employment_rate)),
    metric("工资总额", fmtMoney(record.labor?.total_wage_gross)),
    metric("家庭消费", fmtMoney(record.market?.household_purchase_total)),
    metric("政府采购", fmtMoney(record.market?.government_procurement_total)),
    metric("总产出", fmtMoney(record.market?.total_output)),
    metric("需求价值", fmtMoney(record.market?.demand_value)),
    metric("劳动份额", fmtPct(record.macro?.labor_share)),
    metric("岗位填充", fmtPct(record.labor?.job_fill_rate)),
  ].join("");
}

function renderStages() {
  const events = state.events.slice(-80);
  el.eventCount.textContent = String(state.events.length);
  el.stageTimeline.innerHTML = events
    .slice(-24)
    .map((event) => {
      const cls = event.status === "error" ? "error" : event.event === "start" ? "active" : "";
      const title = `${event.stage || ""} ${event.event || ""}`;
      return `<div class="stage-chip ${cls}" title="${title}"></div>`;
    })
    .join("");

  if (!events.length) {
    el.stageRows.innerHTML = `<p class="muted-empty">当前 run 没有阶段事件。新仿真会自动写入 stage_events.jsonl。</p>`;
    return;
  }

  el.stageRows.innerHTML = events
    .slice()
    .reverse()
    .map((event) => {
      const tagClass = event.status === "error" ? "error" : event.event === "end" ? "end" : "";
      const duration =
        event.elapsed_seconds !== undefined ? `${fmtNumber(event.elapsed_seconds, 2)}s` : "";
      const time = fmtTime(event.timestamp);
      return `
        <div class="stage-row">
          <span class="stage-tag ${tagClass}">${event.event || "--"}</span>
          <strong title="${event.stage || ""}">${event.stage || "--"}</strong>
          <small>${duration || time}</small>
        </div>
      `;
    })
    .join("");
}

function renderChart() {
  const canvas = el.seriesChart;
  const ctx = canvas.getContext("2d");
  const rect = canvas.getBoundingClientRect();
  const scale = window.devicePixelRatio || 1;
  canvas.width = Math.max(320, Math.floor(rect.width * scale));
  canvas.height = Math.floor(220 * scale);
  ctx.setTransform(scale, 0, 0, scale, 0, 0);
  ctx.clearRect(0, 0, rect.width, 220);

  const records = state.records;
  if (records.length < 2) {
    drawEmptyChart(ctx, rect.width, 220);
    return;
  }

  const series = [
    { name: "GDP", color: "#0f766e", values: records.map((r) => r.macro?.nominal_gdp || 0) },
    { name: "工资", color: "#2563eb", values: records.map((r) => r.labor?.total_wage_gross || 0) },
    { name: "消费", color: "#b7791f", values: records.map((r) => r.market?.household_purchase_total || 0) },
  ];
  const allValues = series.flatMap((item) => item.values).filter((num) => Number.isFinite(num));
  const min = Math.min(0, ...allValues);
  const max = Math.max(1, ...allValues);
  const pad = { left: 48, top: 16, right: 16, bottom: 34 };
  const w = rect.width - pad.left - pad.right;
  const h = 220 - pad.top - pad.bottom;

  ctx.strokeStyle = "#d8e0dd";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top);
  ctx.lineTo(pad.left, pad.top + h);
  ctx.lineTo(pad.left + w, pad.top + h);
  ctx.stroke();

  ctx.fillStyle = "#65716e";
  ctx.font = "12px sans-serif";
  ctx.fillText(fmtMoney(max), 4, pad.top + 10);
  ctx.fillText(fmtMoney(min), 4, pad.top + h);

  const xFor = (i) => pad.left + (records.length === 1 ? 0 : (i / (records.length - 1)) * w);
  const yFor = (value) => pad.top + h - ((value - min) / (max - min)) * h;

  for (const item of series) {
    ctx.strokeStyle = item.color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    item.values.forEach((value, index) => {
      const x = xFor(index);
      const y = yFor(value);
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  const markerX = xFor(state.replayIndex);
  ctx.strokeStyle = "#17211f";
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(markerX, pad.top);
  ctx.lineTo(markerX, pad.top + h);
  ctx.stroke();
  ctx.setLineDash([]);

  let legendX = pad.left;
  for (const item of series) {
    ctx.fillStyle = item.color;
    ctx.fillRect(legendX, 204, 10, 10);
    ctx.fillStyle = "#17211f";
    ctx.fillText(item.name, legendX + 16, 214);
    legendX += 74;
  }
}

function drawEmptyChart(ctx, width, height) {
  ctx.fillStyle = "#65716e";
  ctx.font = "13px sans-serif";
  ctx.fillText("需要至少两个月度记录绘制路径", 14, 30);
  ctx.strokeStyle = "#d8e0dd";
  ctx.beginPath();
  ctx.moveTo(14, height - 28);
  ctx.lineTo(width - 14, height - 28);
  ctx.stroke();
}

function clearChart() {
  const ctx = el.seriesChart.getContext("2d");
  ctx.clearRect(0, 0, el.seriesChart.width, el.seriesChart.height);
}

function renderDetails(record) {
  if (!record) {
    el.profitGrid.innerHTML = "";
    el.creditGrid.innerHTML = "";
    el.accountingGrid.innerHTML = "";
    el.gapList.innerHTML = "";
    return;
  }

  const pressure = record.profit_pressure || {};
  el.profitState.textContent = pressure.sales_gap_to_wages > 0 ? "gap" : "closed";
  el.profitGrid.innerHTML = [
    detail("企业收入", fmtMoney(pressure.realized_income)),
    detail("企业支出", fmtMoney(pressure.realized_expenses)),
    detail("企业利润", fmtMoney(pressure.realized_profit)),
    detail("工资支出", fmtMoney(pressure.wage_expense)),
    detail("销售工资缺口", fmtMoney(pressure.sales_gap_to_wages)),
    detail("收入/工资", fmtNumber(pressure.income_to_wage_ratio, 2)),
  ].join("");

  const gaps = pressure.top_sales_gap_to_wages || [];
  el.gapList.innerHTML = gaps.length
    ? gaps
        .slice(0, 6)
        .map(
          (item) => `
            <div class="gap-row">
              <span title="${item.firm_id || ""}">${item.firm_id || "--"}</span>
              <strong>${fmtMoney(item.sales_gap_to_wages)}</strong>
            </div>
          `
        )
        .join("")
    : `<p class="muted-empty">无企业工资销售缺口排行。</p>`;

  const credit = record.firm_credit || {};
  el.creditGrid.innerHTML = [
    detail("利息合计", fmtMoney(credit.interest_total)),
    detail("还款合计", fmtMoney(credit.repayment_total)),
    detail("违约企业", fmtNumber(credit.defaulted_count)),
    detail("有信贷记录企业", fmtNumber(credit.firm_count)),
  ].join("");

  const accounting = record.accounting || {};
  el.accountingGrid.innerHTML = [
    detail("不变量", accounting.ok ? "OK" : "异常"),
    detail("错误数", fmtNumber(accounting.error_count)),
    detail("警告数", fmtNumber(accounting.warning_count)),
    detail("现金账本", fmtMoney(accounting.ledger_total_cash)),
    detail("负现金企业", fmtNumber(accounting.negative_firm_cash_count)),
    detail("流量残差", fmtNumber(accounting.flow_residual, 6)),
  ].join("");
}

function stepReplay(delta) {
  if (!state.records.length) return;
  const max = state.records.length - 1;
  state.replayIndex = Math.max(0, Math.min(max, state.replayIndex + delta));
  renderAll();
}

function togglePlay() {
  if (state.playTimer) {
    clearInterval(state.playTimer);
    state.playTimer = null;
    el.playBtn.textContent = "播放";
    return;
  }
  el.playBtn.textContent = "暂停";
  state.playTimer = setInterval(() => {
    if (!state.records.length) return;
    if (state.replayIndex >= state.records.length - 1) {
      state.replayIndex = 0;
    } else {
      state.replayIndex += 1;
    }
    renderAll();
  }, Number(el.speedSelect.value || 1000));
}

el.runSelect.addEventListener("change", async () => {
  if (!el.runSelect.value) return;
  state.replayIndex = 0;
  await loadRun(el.runSelect.value);
});

el.refreshBtn.addEventListener("click", () => loadRuns(true).catch(showError));
el.prevBtn.addEventListener("click", () => stepReplay(-1));
el.nextBtn.addEventListener("click", () => stepReplay(1));
el.playBtn.addEventListener("click", togglePlay);
el.speedSelect.addEventListener("change", () => {
  if (state.playTimer) {
    togglePlay();
    togglePlay();
  }
});
el.monthRange.addEventListener("input", () => {
  state.replayIndex = Number(el.monthRange.value || 0);
  renderAll();
});
window.addEventListener("resize", () => renderChart());

function showError(error) {
  el.activeStage.textContent = `加载失败: ${error.message}`;
  el.liveDot.className = "live-dot error";
}

loadRuns(false).catch(showError);
setInterval(async () => {
  try {
    await loadRuns(true);
  } catch (error) {
    showError(error);
  }
}, 2500);

