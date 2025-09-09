
import time
import io
import math
import random
import pandas as pd
import streamlit as st
from ortools.sat.python import cp_model

# ------------------------------
# Page setup (logic unchanged; UI only)
# ------------------------------
st.set_page_config(page_title="교회 매칭 프로그램", layout="wide")
st.markdown('<div style="text-align:center;margin:6px 0 10px 0;"><h1 style="margin:0;font-weight:800;color:#2e5a88;">교회 매칭 프로그램</h1></div>', unsafe_allow_html=True)
st.markdown('<div style="text-align:center;color:#7a8ca3;margin-bottom:18px;">팀 번호 + 이름(가나다순, “/” 구분) • 균형 매칭</div>', unsafe_allow_html=True)

# ------------------------------
# 가나다순 정렬 키 (logic unchanged)
# ------------------------------
BASE, CHOS, JUNG = 0xAC00, 588, 28
def hangul_key(s: str):
    ks = []
    for ch in str(s):
        o = ord(ch)
        if 0xAC00 <= o <= 0xD7A3:
            sidx = o - BASE
            cho = sidx // CHOS
            jung = (sidx % CHOS) // JUNG
            jong = sidx % JUNG
            ks.append((0, cho, jung, jong))
        else:
            ks.append((1, o))
    return tuple(ks)

# ------------------------------
# 전처리 유틸 (logic unchanged)
# ------------------------------
AGE_BANDS = ["10대","20대","30대","40대","50대","60대","70대"]

def age_to_band(age: int) -> str:
    try:
        a = int(age)
    except Exception:
        return None
    if a < 20:  return "10대"
    if a < 30:  return "20대"
    if a < 40:  return "30대"
    if a < 50:  return "40대"
    if a < 60:  return "50대"
    if a < 70:  return "60대"
    return "70대"

def normalize_gender(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s in ["남","남자","M","m","male","Male"]:
        return "남"
    if s in ["여","여자","F","f","female","Female"]:
        return "여"
    return None

# ------------------------------
# 팀 크기 결정 (logic unchanged)
# 7 기본, 6/8 허용(합계 ≤4), 7에 근접, 6/8 균형
# ------------------------------
def choose_group_sizes(N: int, max_offsize: int = 4):
    best = None
    target_T = int(round(N/7))
    for x6 in range(0, max_offsize+1):
        for x8 in range(0, max_offsize - x6 + 1):
            rem = N - (6*x6 + 8*x8)
            if rem < 0:
                continue
            if rem % 7 != 0:
                continue
            x7 = rem // 7
            T = x6 + x7 + x8
            off = x6 + x8
            score = (abs(T - target_T), off, abs(x8 - x6))
            cand = (score, x6, x7, x8)
            if best is None or cand < best:
                best = cand
    if best is None:
        return None, f"해결 실패: 6/7/8인 조의 조합으로 총원 {N}명을 구성할 수 없습니다."
    else:
        (_, x6, x7, x8) = best
        sizes = [6]*x6 + [7]*x7 + [8]*x8
        return sizes, None

def allowed_male_bounds(size: int):
    if size == 7: return 3,4
    if size == 6: return 2,4
    if size == 8: return 3,5
    lo = int(math.floor(0.4*size))
    hi = int(math.ceil(0.6*size))
    return lo, hi

# ------------------------------
# 핵심 솔버 (logic unchanged)
# - 교회: 기본 ≤2/팀, 불가 시 3·4만 정확히 필요한 만큼 (4는 약간 불리)
# - 나이대: 기본 ≤2/팀, 불가 시 3만 정확히 필요한 만큼
# - 성비: 범위 슬랙 최소화
# ------------------------------
def solve_assignment(df, seed=0, time_limit=10, max_per_church=4):
    people = df.to_dict('records')
    N = len(people)
    sizes, warn = choose_group_sizes(N, max_offsize=4)
    if sizes is None:
        return None, None, "조 크기 계산 실패", None
    G = len(sizes)

    males = [i for i,p in enumerate(people) if p['성별'] == '남']

    churches = sorted(df['교회 이름'].fillna("미상").astype(str).unique().tolist())
    church_members = {c: [i for i,p in enumerate(people) if str(p['교회 이름']) == c] for c in churches}
    bands = AGE_BANDS
    band_members = {b: [i for i,p in enumerate(people) if p['나이대'] == b] for b in bands}

    # 사전 타당성
    for c, members in church_members.items():
        if len(members) > max_per_church*G:
            return None, None, f"불가능: {c} 인원 {len(members)} > 허용 {max_per_church*G}", None
    for b, members in band_members.items():
        if len(members) > 3*G:  # 나이대는 기본 2, 불가시 3 허용
            return None, None, f"불가능: {b} 인원 {len(members)} > 허용 {3*G}", None

    # 초과 필요 계산
    church_counts = {c: len(members) for c, members in church_members.items()}
    extra_needed = {c: max(0, cnt - 2*G) for c, cnt in church_counts.items()}
    age_counts = {b: len(members) for b, members in band_members.items()}
    age_extra_needed = {b: max(0, cnt - 2*G) for b, cnt in age_counts.items()}

    model = cp_model.CpModel()

    # 배정 변수
    x = {}
    for i in range(N):
        for g in range(G):
            x[(i,g)] = model.NewBoolVar(f"x_{i}_{g}")

    # 각 사람 정확히 1팀
    for i in range(N):
        model.Add(sum(x[(i,g)] for g in range(G)) == 1)

    # 팀 크기 고정
    for g in range(G):
        model.Add(sum(x[(i,g)] for i in range(N)) == sizes[g])

    # 성비 제약(슬랙)
    sL = []
    sU = []
    for g in range(G):
        mc = model.NewIntVar(0, sizes[g], f"male_{g}")
        model.Add(mc == sum(x[(i,g)] for i in males))
        lo, hi = allowed_male_bounds(sizes[g])
        sl = model.NewIntVar(0, sizes[g], f"sL_{g}")
        su = model.NewIntVar(0, sizes[g], f"sU_{g}")
        model.Add(mc >= lo - sl)
        model.Add(mc <= hi + su)
        sL.append(sl)
        sU.append(su)

    # 동일 교회: 기본 ≤2, 불가 시 3·4만 정확히 필요한 만큼
    zero = model.NewIntVar(0, 0, "zero_const")
    b4_flags = []        # 팀 내 동일 교회 4명 케이스 표시
    shortfall_church = {}  # 초과 충족 부족분(벌점용)
    for c in churches:
        y_vars = []
        members = church_members[c]
        for g in range(G):
            cnt = model.NewIntVar(0, min(max_per_church, len(members)), f"church_{c}_{g}")
            model.Add(cnt == sum(x[(i,g)] for i in members))
            model.Add(cnt <= max_per_church)  # ≤4(하드)

            t = model.NewIntVar(-2, max_per_church-2, f"t_{c}_{g}")
            model.Add(t == cnt - 2)
            y = model.NewIntVar(0, 2, f"y_{c}_{g}")  # {0,1,2} → 2는 4명
            model.AddMaxEquality(y, [t, zero])
            y_vars.append(y)

            b4 = model.NewBoolVar(f"b4_{c}_{g}")
            model.Add(y >= 2).OnlyEnforceIf(b4)
            model.Add(y <= 2).OnlyEnforceIf(b4)
            model.Add(y <= 1).OnlyEnforceIf(b4.Not())
            b4_flags.append(b4)

        s_c = model.NewIntVar(0, int(extra_needed[c]), f"short_c_{c}")
        shortfall_church[c] = s_c
        model.Add(sum(y_vars) + s_c == int(extra_needed[c]))

    # 동일 나이대: 기본 ≤2, 불가 시 3만 정확히 필요한 만큼
    age_is3_flags = []
    age_shortfall = {}
    for b in bands:
        y_vars = []
        members = band_members[b]
        for g in range(G):
            cnt = model.NewIntVar(0, min(3, len(members)), f"band_{b}_{g}")
            model.Add(cnt == sum(x[(i,g)] for i in members))
            model.Add(cnt <= 3)
            is3 = model.NewBoolVar(f"is3_band_{b}_{g}")
            model.Add(cnt == 3).OnlyEnforceIf(is3)
            model.Add(cnt != 3).OnlyEnforceIf(is3.Not())
            age_is3_flags.append(is3)
            y_vars.append(is3)
        s_b = model.NewIntVar(0, int(age_extra_needed[b]), f"short_b_{b}")
        age_shortfall[b] = s_b
        model.Add(sum(y_vars) + s_b == int(age_extra_needed[b]))

    # 목적함수
    rand = random.Random(int(time.time()) % (10**6))
    noise_terms = []
    for i in range(N):
        for g in range(G):
            w = rand.randint(0, 2)
            if w > 0:
                noise_terms.append(w * x[(i,g)])

    model.Minimize(
        5000 * sum(shortfall_church.values()) + 5000 * sum(age_shortfall.values()) +
        1000 * sum(sL) + 1000 * sum(sU) +
        5 * sum(b4_flags) +
        1 * sum(age_is3_flags) +
        1 * sum(noise_terms)
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = 8
    try:
        solver.parameters.random_seed = int(seed)
    except Exception:
        pass

    res = solver.Solve(model)
    if res not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, None, "해결 실패: 제약을 만족하는 해를 찾지 못했습니다. (시간 제한/분포 문제 가능)", None

    groups = []
    for g in range(G):
        members = [i for i in range(N) if solver.Value(x[(i,g)]) == 1]
        groups.append(members)

    total_slack = int(sum(solver.Value(v) for v in sL) + sum(solver.Value(v) for v in sU))
    warn_list = []
    if total_slack > 0:
        warn_list.append(f"주의: 성비 제약을 {total_slack}명만큼 완화하여 해를 구성했습니다.")

    return groups, warn_list, None, sizes

# ------------------------------
# UI Helper (new): 선물꾸러미 카드 + 이름칩 (디자인만, 로직 불변)
# ------------------------------
def render_gift(team_no: int, names_line: str, title_px: int, names_px: int):
    chips = [n.strip() for n in names_line.split('/') if n.strip()]
    gift_html = [
        '<div style="background: rgba(255,255,255,0.78); border:1px solid rgba(46,90,136,0.10);'
        ' border-radius: 20px; box-shadow: 0 10px 26px rgba(46,90,136,0.10); padding: 18px 18px 14px; margin-bottom: 14px;">',
        '  <div style="display:flex; justify-content:center; align-items:center; gap:8px; margin-bottom: 10px;">',
        '    <span style="width:10px;height:10px;border-radius:999px;background:#3a7ca5;box-shadow:0 0 0 3px rgba(58,124,165,0.15) inset;"></span>',
        f'    <div style="font-weight:800;color:#2e5a88;font-size:{title_px}px;">팀 {team_no}</div>',
        '    <span style="width:10px;height:10px;border-radius:999px;background:#3a7ca5;box-shadow:0 0 0 3px rgba(58,124,165,0.15) inset;"></span>',
        '  </div>',
        '  <div style="display:flex;flex-wrap:wrap;gap:8px;justify-content:center;">',
    ]
    for c in chips:
        gift_html.append(
            f'    <span style="background:#fff;border:1px solid rgba(46,90,136,0.15);border-radius:12px;'
            f'padding:6px 10px;font-weight:600;color:#1b365d;box-shadow:0 4px 10px rgba(46,90,136,0.08);'
            f'font-size:{names_px}px;">{c}</span>'
        )
    gift_html.append('  </div>')
    gift_html.append('</div>')
    return '\\n'.join(gift_html)

# ------------------------------
# Sidebar (디자인 슬라이더만; 로직 불변)
# ------------------------------
with st.sidebar:
    st.header("설정")
    uploaded = st.file_uploader("엑셀 업로드 (.xlsx)", type=["xlsx"])
    time_limit = st.slider("해결 시간 제한(초)", min_value=5, max_value=30, value=10, step=1)
    title_px = st.slider("제목 글자 크기(px)", 48, 96, 64, 2)
    names_px = st.slider("이름 글자 크기(px)", 24, 64, 36, 2)
    run_btn = st.button("🎲 매칭 시작")

st.markdown('필수 컬럼: `이름`, `성별(남/여)`, `교회 이름`, `나이` · 결과는 **팀 번호 + 이름(가나다순, `/` 구분)** 만 표시됩니다.', unsafe_allow_html=True)

# ------------------------------
# Data load & validate (logic unchanged)
# ------------------------------
df = None
if uploaded is not None:
    try:
        df = pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"엑셀 읽기 오류: {e}")

if df is not None:
    required = ["이름","성별","교회 이름","나이"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"필수 컬럼 누락: {missing}")
        st.stop()

    df = df.copy()
    df["성별"] = df["성별"].apply(normalize_gender)
    if df["성별"].isna().any():
        st.error("성별 값 표준화 실패 행이 있습니다. ('남'/'여'만 허용)")
        st.dataframe(df[df["성별"].isna()])
        st.stop()

    df["나이대"] = df["나이"].apply(age_to_band)
    if df["나이대"].isna().any():
        st.error("나이 → 나이대 변환 실패 행이 있습니다. (정수 나이 필요)")
        st.dataframe(df[df["나이대"].isna()])
        st.stop()

    N = len(df)
    sizes, warn = choose_group_sizes(N, max_offsize=4)
    if sizes is None:
        st.error(warn)
        st.stop()
    if warn:
        st.warning(warn)

    if run_btn:
        ph = st.empty()
        for pct in range(0, 101, 7):
            ph.progress(pct, text="배치 탐색 중...")
            time.sleep(0.03)

        groups, warn_list, err, sizes = solve_assignment(df, time_limit=time_limit, max_per_church=4)

        if err:
            st.error(err)
            st.stop()
        if warn_list:
            for w in warn_list:
                st.warning(w)

        people = df.to_dict('records')

        # Prepare names per team (ga-na-da order, " / " join) — logic unchanged
        names_per_team = []
        for g, members in enumerate(groups):
            team_names = [people[i]['이름'] for i in members]
            team_names_sorted = sorted(team_names, key=hangul_key)
            names_per_team.append(" / ".join(team_names_sorted))

        # Initialize session state (viewer) — logic unchanged
        st.session_state.assignment_ready = True
        st.session_state.names_per_team = names_per_team
        st.session_state.team_count = len(names_per_team)
        st.session_state.team_idx = 0
        st.session_state.final_view = False

# ------------------------------
# Viewer (디자인만 gift-card; 로직 불변)
# ------------------------------
if st.session_state.get("assignment_ready", False):
    # Toolbar
    c1, c2, c3, c4 = st.columns([1,1,1,1])
    with c1:
        if st.button("◀ 이전 팀"):
            st.session_state.team_idx = (st.session_state.team_idx - 1) % st.session_state.team_count
            st.session_state.final_view = False
    with c2:
        if st.button("최종 결과 보기"):
            st.session_state.final_view = True
    with c3:
        st.markdown(f'<div style="display:flex;justify-content:center;"><span style="font-weight:600;padding:6px 12px;border-radius:999px;border:1px solid rgba(46,90,136,0.25);background:#fff;">{st.session_state.team_idx+1} / {st.session_state.team_count}팀</span></div>', unsafe_allow_html=True)
    with c4:
        if st.button("다음 팀 ▶"):
            if st.session_state.team_idx < st.session_state.team_count - 1:
                st.session_state.team_idx += 1
                st.session_state.final_view = False
            else:
                st.session_state.final_view = True

    # Content
    if st.session_state.final_view:
        st.markdown('<div style="text-align:center;font-weight:800;color:#2e5a88;margin:10px 0 8px 0;">최종 결과</div>', unsafe_allow_html=True)
        cols = st.columns(2)
        for idx, names_line_tmp in enumerate(st.session_state.names_per_team, start=1):
            with cols[(idx-1) % 2]:
                st.markdown(render_gift(idx, names_line_tmp, title_px, names_px), unsafe_allow_html=True)
    else:
        cur_idx = st.session_state.team_idx
        st.markdown(render_gift(cur_idx+1, st.session_state.names_per_team[cur_idx], title_px, names_px), unsafe_allow_html=True)

    # Download (teams + names only)
    rows = []
    for g, names_line_tmp in enumerate(st.session_state.names_per_team):
        for name in names_line_tmp.split(" / "):
            rows.append({"팀": g+1, "이름": name})
    out_df = pd.DataFrame(rows)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        out_df.to_excel(writer, index=False, sheet_name="TeamsOnly")
    st.download_button("결과 엑셀 다운로드(팀+이름, 가나다순)", data=buf.getvalue(),
                       file_name="teams_names_only.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ------------------------------
# 하단 진단 요약 (축소)
# ------------------------------
if st.session_state.get("assignment_ready", False):
    with st.expander("진단 요약 보기", expanded=False):
        try:
            G = st.session_state.team_count
            st.write(f"팀 수: {G}")
            st.write('팀 크기(정렬): ' + ', '.join(map(str, sorted(sizes))))
            church_counts = df['교회 이름'].fillna('미상').astype(str).str.strip().value_counts().rename_axis('교회').reset_index(name='인원')
            church_counts['초과필요(z합)'] = (church_counts['인원'] - 2*G).clip(lower=0)
            st.dataframe(church_counts, use_container_width=True)
            age_counts = df['나이대'].value_counts().rename_axis('나이대').reset_index(name='인원').sort_values('나이대')
            age_counts['초과필요(3인팀수)'] = (age_counts['인원'] - 2*G).clip(lower=0)
            st.dataframe(age_counts, use_container_width=True)
        except Exception:
            pass
