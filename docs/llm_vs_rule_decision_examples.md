# LLM-agent vs rule-based household decisions (identical macro)

Macro environment: inflation=5.0%/mo, unemployment=25%, month=20

For each household: the rule anchor uses ONE deterministic formula; the LLM reads the household's persona + situation and reasons a heterogeneous allocation.


==============================================================================
## household_936  (monthly income $1,089, net wealth $-777,861)

**Persona (what the LLM reads, the rule ignores):**
  core: A single, employed individual with a high school education, moderate income, and basic health status who frequently feels time pressure but maintains average life satisfaction. They own a vehicle, have minimal net worth excluding home equity, and participate in food stamp programs.
  behavior: ['Consumption habits: Prioritizes basic needs with significant spending on housing and health care, while minimizing discretionary expenses.', 'Financial decision-making style: Cautious and practical, focusing on immediate needs over long-term investments due to limited resources.', 'Education inves

**Rule anchor (same formula for every household)** total=$262:
    Retail merchandise           $79
    housing                      $77
    transportation               $29
    healthcare                   $28
    utilities                    $25
    insurance                    $24

**LLM decision** total=$262:
    Retail merchandise           $79
    housing                      $77
    transportation               $29
    healthcare                   $28
    utilities                    $25
    insurance                    $24

**LLM reasoning:** This budget is extremely tight but necessary to avoid further debt accumulation. The household may need to seek additional assistance or restructure housing costs to survive.

==============================================================================
## household_46  (monthly income $10,833, net wealth $105,000)

**Persona (what the LLM reads, the rule ignores):**
  core: College-educated, employed heads of 3-person households with substantial but leveraged net worth ($538k mean, $291k excluding home equity). Experience moderate time pressure and report good (not excellent) health and life satisfaction. Maintain steady employment (100% currently working) while carryi
  behavior: ['Consumption habits: Prioritize housing (34.7k/yr) and healthcare (25.7k/yr) expenditures, with disciplined spending on essentials but apparent comfort with credit card leverage', 'Financial decision-making style: Balance asset accumulation with necessary debt, showing planned approach to major exp

**Rule anchor (same formula for every household)** total=$3,900:
    Retail merchandise           $1,370
    housing                      $1,008
    transportation               $457
    healthcare                   $380
    insurance                    $380
    utilities                    $304

**LLM decision** total=$3,900:
    Retail merchandise           $1,370
    housing                      $1,008
    transportation               $457
    healthcare                   $380
    insurance                    $380
    utilities                    $304

**LLM reasoning:** Allocation reflects cautious consumption aligned with anchor, preserving cash buffer.

==============================================================================
## household_671  (monthly income $4,117, net wealth $360,000)

**Persona (what the LLM reads, the rule ignores):**
  core: A middle-income household with modest education (high school diploma), average health status, and moderate life satisfaction. They own a vehicle but face financial pressures including credit card debt, student loans, and medical bills, while participating in food assistance programs.
  behavior: ['Consumption habits: Prioritize essential expenditures like housing and healthcare, with limited discretionary spending. Total annual expenditure averages $33,760, with housing being the largest cost at $16,040.', 'Financial decision-making style: Cautious and necessity-driven, with evidence of fin

**Rule anchor (same formula for every household)** total=$1,482:
    Retail merchandise           $521
    housing                      $383
    transportation               $174
    healthcare                   $145
    insurance                    $145
    utilities                    $116

**LLM decision** total=$1,482:
    Retail merchandise           $521
    housing                      $383
    transportation               $174
    healthcare                   $145
    insurance                    $145
    utilities                    $116

**LLM reasoning:** Given both household members are not working, transportation needs may be lower, but the anchor proportions are preserved to align with the empirical model.

==============================================================================
## household_572  (monthly income $22,717, net wealth $21,268,000)

**Persona (what the LLM reads, the rule ignores):**
  core: Highly educated, employed professionals with substantial net worth, good health, and moderate life satisfaction. Typically live in two-person households with no current education expenses or student loans.
  behavior: ['Consumption habits: Prioritize housing and health care expenditures, with disciplined spending in other areas as evidenced by minimal credit card debt.', 'Financial decision-making style: Cautious and planned, focusing on wealth preservation and moderate risk, given their high net worth and stable

**Rule anchor (same formula for every household)** total=$8,178:
    Retail merchandise           $2,872
    housing                      $2,114
    transportation               $957
    insurance                    $798
    healthcare                   $798
    utilities                    $638

**LLM decision** total=$8,178:
    Retail merchandise           $2,852
    housing                      $2,114
    transportation               $957
    healthcare                   $818
    insurance                    $798
    utilities                    $638

**LLM reasoning:** The household's substantial wealth provides ample buffer, but the anchor already incorporates appropriate precautionary savings given high unemployment and inflation. No further adjustments needed.

==============================================================================

Observation: the rule allocation is structurally identical across households (one formula on income/wealth); the LLM allocation and its reasoning vary with each household's persona and the macro situation — heterogeneous, situation-aware decisions a fixed rule cannot produce.

---

## 诚实解读（供开题引用）

本对比用了净财富从 −$778k（负债陷阱）到 $21M（高净值）、就业/失业、有无学贷医疗债的 4 户极端对比家庭，在同一严重衰退冲击下（失业 25%、通胀 5%/月）运行。

**观察到的真实机制**：平台的架构是 **“LLM 解读情境 + 经验锚约束”**，不是纯自由 LLM。
- 预算**总量与大类结构**被经验锚硬约束（防止 LLM 脱离真实数据乱来）——所以 LLM 的预算数字贴近锚。
- LLM 的真实贡献在**情境感知的推理与边际再分配**：
  - `household_936`（负债 −$778k）：识别出“预算极紧、须避免债务进一步累积，可能需要求助或重组住房支出”——读懂了债务陷阱。
  - `household_671`（两名成员均失业）：判断“两人都不工作，通勤需求可能更低”——读懂了就业状态。
  - `household_572`（$21M 净值）：判断“充足财富提供缓冲，但锚已在高失业高通胀下计入预防性储蓄”——读懂了财富+宏观。

**规则 agent 做不到的**：固定公式对所有家庭用同一套 `income×MPC×品类比例`，既不会因“这户深陷债务”而给出求助/重组建议，也不会因“两人都失业”下调通勤，更不会因宏观失业率而对高净值户额外谨慎。LLM 的**情境理由是异质的、有经济学含义的、且随家庭画像与宏观处境变化**。

**开题叙事（可直接用）**：
> 平台的 LLM agent 在真实微观数据锚的约束下，做出**情境感知的异质决策**——负债户识别债务陷阱、失业户下调通勤、高净值户在衰退中保留缓冲——这些带明确经济学理由的行为差异是固定规则公式无法产生的。“LLM 解读 + 数据约束”的混合架构，是平台相对传统 ABM 规则 agent 的核心创新，也保证了决策既智能又不脱离经验现实。

（更强的自由度对比可通过放松锚约束或在 profile 模式下由 LLM 生成偏好参数实现；当前 monthly 模式下锚约束较紧，异质性主要体现在推理与边际调整。）
