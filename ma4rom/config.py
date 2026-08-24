
import os
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


#  ontology  → input/<CURRENT_DATABASE>/ontology.ttl
#  输出目录   → output/<CURRENT_DATABASE>/
#  最终映射   → output/<CURRENT_DATABASE>/rodi_<CURRENT_DATABASE>_generated_mapping.ttl
#
# RODI 基础实验/消融实验常用数据库名，复制到 CURRENT_DATABASE 即可切换：
# renamed 系列：
#   cmt_renamed
#   conference_renamed
#   sigkdd_renamed
# structured 系列：
#   cmt_structured
#   conference_structured
#   sigkdd_structured
# mixed / combined case：
#   sigkdd_mixed
# missing FK：
#   conference_nofks
# denormalized：
#   cmt_denormalized
# geodata：
#   mondial_rel
# oil & gas：
#   npd_atomic_tests

DEFAULT_DATABASE = "cmt_structured"  # 默认单场景示例数据库
CURRENT_DATABASE = os.environ.get("MAMG_CURRENT_DATABASE", DEFAULT_DATABASE)  # 批量实验时可由环境变量临时覆盖

# 统一入口总开关。可用 MAMG_IS_ABLATION=true 临时开启，无需改源码。
isAblation = _env_bool("MAMG_IS_ABLATION", False)

# 超参数实验总开关。
# 默认关闭，保证 `python ma4rom/main.py` 运行单次标准流程；
# 如需复现实验中的超参数扫描，可显式设置 MAMG_IS_HYPER=true。
isHyper = _env_bool("MAMG_IS_HYPER", False) and not isAblation

# HyperSelect 用字符串控制当前进行哪组实验：
#   "dp_weight"                  -> DP 候选打分权重
#   "dp_confidence"              -> DP 置信度阈值
#   "fk_completion_ind_threshold"-> FK_COMPLETION_IND_THRESHOLD
HyperSelect = os.environ.get("MAMG_HYPER_SELECT", "fk_completion_ind_threshold").strip()

# 单个挡位名称。父进程串行调度时会为每个子进程注入该值。
HyperLevel = os.environ.get("MAMG_HYPER_LEVEL", "").strip()

# 子进程标记：由 main.py 的超参数调度器设置，避免递归启动。
HyperChild = _env_bool("MAMG_HYPER_CHILD", False)

# 每组实验允许的挡位，文件夹名也直接复用这些 level 名称。
HYPER_LEVELS_BY_SELECT = {
    "dp_weight": ["w_0p9_0p1", "w_0p7_0p3", "w_0p5_0p5", "w_0p3_0p7"],
    "dp_confidence": ["loose", "default", "strict"],
    "fk_completion_ind_threshold": ["0p6", "0p8", "0p9", "0p95", "1p0"],
}


def _default_hyper_level(select_name: str) -> str:
    levels = HYPER_LEVELS_BY_SELECT.get(select_name, [])
    return levels[0] if levels else ""


if isHyper and not HyperLevel:
    HyperLevel = _default_hyper_level(HyperSelect)

#  PostgreSQL 连接配置

DB_CONFIG = {
    "host":     os.environ.get("PGHOST", "localhost"),
    "port":     int(os.environ.get("PGPORT", "5432")),
    "database": CURRENT_DATABASE,       # 与上方 CURRENT_DATABASE 保持同步
    "user":     os.environ.get("PGUSER", "postgres"),
    # Never persist a database password in source.  Empty means psycopg2 will
    # use the normal PostgreSQL credential mechanisms (for example .pgpass).
    "password": os.environ.get("PGPASSWORD", ""),
}

# PostgreSQLschema名
DB_SCHEMA_BY_DATABASE = {
    "cmt_denormalized_-100": "cmt_denormalized",
    "cmt_renamed_-100": "cmt_renamed",
    "cmt_structured_-100": "cmt_structured",
    "conference_naive_-100": "conference_naive",
    "conference_renamed_-100": "conference_renamed",
    "conference_structured_-100": "conference_structured",
    "mondial_rel": "mondial_rdf2sql_standard",
    "mondial_rel_-100": "mondial_rdf2sql_standard",
    "npd_atomic_tests_-100": "npd_atomic_tests",
    "sigkdd_mixed_-100": "sigkdd_mixed",
    "sigkdd_naive_-100": "sigkdd_naive",
    "sigkdd_renamed_-100": "sigkdd_renamed",
    "sigkdd_structured_-100": "sigkdd_structured",
}
DB_SCHEMA_NAME = os.environ.get(
    "MAMG_DB_SCHEMA",
    DB_SCHEMA_BY_DATABASE.get(CURRENT_DATABASE, "public"),
)

# Credentials are injected only through the process environment.  This keeps
# experimental runs reproducible without persisting a personal API key.
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com")
_requested_llm_model = os.environ.get("LLM_MODEL", "deepseek-v4-flash").strip()
if _requested_llm_model != "deepseek-v4-flash":
    raise ValueError(
        "This camera-ready repair path only permits LLM_MODEL=deepseek-v4-flash"
    )
LLM_MODEL = "deepseek-v4-flash"
# Fresh validation is deliberately single-model.  A fallback model can turn a
# transient error into an unbounded second stream of paid requests and makes
# the reported counters hard to audit.  An explicit opt-in is still possible
# for a separate experiment, but the normal camera-ready path has no fallback.
LLM_FALLBACK_MODELS = []
# DeepSeek V4 accepts this switch through the OpenAI-compatible ``extra_body``
# field.  Keep it off by default: the mapping prompts require a short JSON
# decision, not a reasoning trace.
LLM_THINKING_ENABLED = _env_bool("LLM_THINKING_ENABLED", False)
LLM_MAX_OUTPUT_TOKENS = int(os.environ.get("LLM_MAX_OUTPUT_TOKENS", "512"))
# Hard process-level spend guards.  They are checked before every request and
# raise a distinct exception; callers must not silently retry after the guard.
LLM_MAX_API_ATTEMPTS = int(os.environ.get("LLM_MAX_API_ATTEMPTS", "250"))
LLM_MAX_TOTAL_TOKENS = int(os.environ.get("LLM_MAX_TOTAL_TOKENS", "1500000"))
LLM_TIMEOUT_SECONDS = int(os.environ.get("LLM_TIMEOUT_SECONDS", "45"))
# One initial request plus two local retries before a mapping is marked failed.
LLM_MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", "3"))
LLM_HARD_TIMEOUT_SECONDS = int(
    os.environ.get("LLM_HARD_TIMEOUT_SECONDS", str(LLM_TIMEOUT_SECONDS + 10))
)

_PROJECT_ROOT = Path(__file__).resolve().parent  # 项目根目录
ONTOLOGY_PATH = str(_PROJECT_ROOT / "input" / CURRENT_DATABASE / "ontology.ttl")  # 当前数据库对应的本体文件路径
_OUTPUT_NAMESPACE = os.environ.get("MAMG_OUTPUT_NAMESPACE", "").strip().strip("/")
if isHyper:
    # 超参数实验输出统一落到 output/hyper/<hyper_name>/<level>/<database>/
    OUTPUT_DIR = str(_PROJECT_ROOT / "output" / "hyper" / HyperSelect / HyperLevel / CURRENT_DATABASE)
elif _OUTPUT_NAMESPACE:
    OUTPUT_DIR = str(_PROJECT_ROOT / "output" / _OUTPUT_NAMESPACE / CURRENT_DATABASE)
else:
    OUTPUT_DIR = str(_PROJECT_ROOT / "output" / CURRENT_DATABASE)                 # 当前数据库对应的输出目录

# 最终 R2RML 映射文件名
OUTPUT_MAPPING_FILENAME = f"rodi_{CURRENT_DATABASE}_generated_mapping.ttl"

# 生成映射时使用的数据资源 URI 前缀（不同数据集可按需切换）
MAPPING_BASE_URL_BY_DB = {
    "npd_atomic_tests": "http://sws.ifi.uio.no/data/npd-v2/",  # NPD 数据集官方数据资源 URI 前缀
}
MAPPING_BASE_URL = MAPPING_BASE_URL_BY_DB.get(CURRENT_DATABASE, "http://example.com/")  # 当前数据库最终使用的资源 URI 前缀

# 是否启用IND外键补全；可用 MAMG_USE_IND_FK_COMPLETION=true 做公平补全实验
USE_IND_FK_COMPLETION = _env_bool("MAMG_USE_IND_FK_COMPLETION", False)
# 当数据库物理FK数量为0时，是否自动启用IND补全
AUTO_ENABLE_IND_ON_NOFKS = True
# 当数据库物理FK较少时，是否自动启用IND补全
AUTO_ENABLE_IND_ON_SPARSE_FKS = True
# “FK较少”的阈值（物理FK数量<=该值时触发）
IND_SPARSE_FK_THRESHOLD = 5

# 是否合并classify_agent推断的LLM外键
USE_LLM_FK_MERGE = False

# OP 映射统一使用等价列发现 + 本体 endpoint 约束 + LLM 选择的新模块。
EQUIV_OP_MIN_ENDPOINT_SCORE = float(os.environ.get("MAMG_EQUIV_OP_MIN_ENDPOINT_SCORE", "0.5"))
EQUIV_OP_LLM_SLEEP_SECONDS = float(os.environ.get("MAMG_EQUIV_OP_LLM_SLEEP", "0.15"))

# 真实值增强全量重判低置信（不做成本裁剪）
MAX_REAL_VALUE_TABLES = -1

# Class / DP 真实值上下文增强策略：
#   none       -> 完全不使用真实数据增强
#   all        -> 所有 Class 和 data_attr 列都使用真实数据增强
#   confidence -> 只增强低置信度 Class / DP（论文默认方法）
CONTEXT_ENHANCEMENT_MODE = os.environ.get(
    "MAMG_CONTEXT_ENHANCEMENT_MODE",
    "confidence",
).strip().lower()
if CONTEXT_ENHANCEMENT_MODE not in {"none", "all", "confidence"}:
    raise ValueError(
        "MAMG_CONTEXT_ENHANCEMENT_MODE 必须是 none / all / confidence，"
        f"当前为 {CONTEXT_ENHANCEMENT_MODE!r}"
    )

# ============================================================
#  Ablation knobs: 将影响结果的阈值/上限统一放到 config
# ============================================================

# ---------- FKCompletion ----------
FK_COMPLETION_EXCLUDED_TYPES = {"boolean", "date", "timestamp", "double precision", "float"}  # IND 外键补全时排除的列类型
FK_COMPLETION_EXCLUDED_COLUMNS = {"type", "status", "kind", "category", "flag", "code"}  # 排除判别/枚举编码列，避免误补 FK
FK_COMPLETION_IND_THRESHOLD = 0.95  # IND 包含依赖阈值：射手列值被目标列覆盖比例达到该值才补成 FK
FK_COMPLETION_ROLE_CLASS_MIN_SCORE = 0.65  # 非 ID 列从列名推断本体角色 Class 的最低名称分
FK_COMPLETION_OP_NAME_MIN_SCORE = 0.55     # 非 ID 列用 ObjectProperty 支撑 FK 时要求的最低属性名匹配分
FK_COMPLETION_STRICT_OP_NAME_MIN_SCORE = 0.80  # 本体+LLM fallback 中要求 OP 名称强匹配，避免蹭到相似但不存在的 OP
FK_COMPLETION_OP_SIDE_MIN_SCORE = 0.50     # 非 ID 列用 ObjectProperty 支撑 FK 时要求 domain/range 至少达到的匹配分
USE_LLM_EMPTY_FK_COMPLETION = True  # 对无非空值的列/空表，使用 LLM 语义补全候选 FK
FK_COMPLETION_LLM_EMPTY_MIN_NAME_SCORE = 0.70  # 空表 LLM 补全后，目标表名/列名语义匹配的最低分
USE_LLM_ONTOLOGY_FK_FALLBACK = True  # IND 召回失败时，基于本体 OP 候选让 LLM 选择空列/空表 FK
FK_COMPLETION_ONTOLOGY_FALLBACK_MIN_SCORE = 0.50  # 本体 OP/role 候选进入 LLM 前的最低语义分

# ---------- classify ----------
CLASSIFY_SAMPLE_ROWS_LIMIT = 5       # Battle 二次裁决时给 LLM 查看每张表的样本行数
CLASSIFY_BATTLE_MATCH_THRESHOLD = 0.8 # Battle 验证 LLM 推断 FK 时要求的最小匹配率

# ---------- Data Property Mapping ----------
DP_MAPPING_CANDIDATE_TEXT_WEIGHT = 0.7            # DP 候选打分中的文本相似度权重 λ_text
DP_MAPPING_CANDIDATE_DOMAIN_WEIGHT = 0.3          # DP 候选打分中的 domain 兼容性权重 λ_dom
DP_MAPPING_CONF_HIGH_TOP1 = 0.7                    # DP 映射候选 top1 分数达到 high 置信的最低分
DP_MAPPING_CONF_HIGH_GAP = 0.2                     # DP 映射 high 置信要求 top1 与 top2 的最小分差
DP_MAPPING_CONF_MEDIUM_TOP1 = 0.4                  # DP 映射候选 top1 分数达到 medium 置信的最低分
DP_MAPPING_CONF_MEDIUM_GAP = 0.1                   # DP 映射 medium 置信要求 top1 与 top2 的最小分差
DP_MAPPING_DATA_ATTR_NULL_FALLBACK_MIN_SCORE = 0.45 # LLM 对数据属性返回 null 时，允许回退 top1 的最低候选分
DP_MAPPING_LOW_CONF_MIN_LOW_COLS = 3                # 触发真实值增强的低置信列数量阈值
DP_MAPPING_LOW_CONF_RATIO_THRESHOLD = 0.25          # 触发真实值增强的低置信列比例阈值
DP_MAPPING_BOOL_CLASS_BASE_WEIGHT = 0.6            # 布尔判别列匹配类型 Class 时，已有候选分的基础权重
DP_MAPPING_BOOL_CLASS_EXACT_BOOST = 1.2            # 布尔列名与 Class 名精确匹配时的提升分

# ---------- RealValue enhancement ----------
REAL_VALUE_SAMPLE_ROWS_LIMIT = 5                         # 真实值增强 重判表/列时读取的样本行数
REAL_VALUE_ENUM_PER_VALUE_LIMIT = 8                      # 真实值增强 处理枚举列时，每个 distinct 值最多抽样的行数
REAL_VALUE_ENUM_MAX_VALUES = 20                          # 真实值增强 处理枚举列时最多查看的 distinct 值数量
REAL_VALUE_BOOL_MAX_VALUES = 5                           # 真实值增强 处理布尔列时最多查看的 distinct 值数量
REAL_VALUE_FK_CONTEXT_MAX_INCOMING = 8                   # 真实值增强 构建 FK 语义上下文时最多保留的 incoming 边数量
REAL_VALUE_ANCESTOR_MAX_DEPTH = 8                        # 真实值增强 使用本体层次结构时向上/向下搜索的最大深度
REAL_VALUE_ENUM_NUMERIC_RATIO_THRESHOLD = 0.4            # 判断枚举值是否更像数值编码的数值比例阈值
REAL_VALUE_ENUM_SAMPLE_DISTINCT_RATIO_THRESHOLD = 0.75   # 判断样本列是否过于离散、难以当作枚举分类列的比例阈值
REAL_VALUE_ENUM_DISTINCT_MAX_FOR_CODE = 10               # 判断编码型枚举列时允许的最大 distinct 数
REAL_VALUE_RULE_STRUCT_SIGNAL_THRESHOLD = 0.22           # 真实值增强 规则侧结构信号达到该值才认为有可用结构证据
REAL_VALUE_RULE_FALLBACK_REPEATED_RATIO = 0.45           # 真实值增强 fallback 判断重复值特征的比例阈值
REAL_VALUE_RULE_FALLBACK_DISTINCT_MAX = 12               # 真实值增强 fallback 判断编码/枚举列时允许的最大 distinct 数
REAL_VALUE_TYPE_HIGH_SCORE = 0.9                         # 真实值增强 类型列重判 high 置信的最低分
REAL_VALUE_TYPE_HIGH_GAP = 0.14                          # 真实值增强 类型列 high 置信要求 top1 与 top2 的最小分差
REAL_VALUE_TYPE_MEDIUM_SCORE = 0.58                      # 真实值增强 类型列重判 medium 置信的最低分
REAL_VALUE_TYPE_MEDIUM_GAP = 0.04                        # 真实值增强 类型列 medium 置信要求 top1 与 top2 的最小分差
REAL_VALUE_TYPE_WEAK_SCORE = 0.52                        # 真实值增强 类型列弱匹配时可接受的最低分
# 单值编码没有可解释的词面语义。只有一个已知 SH 子表在足够多实体上近乎
# 完全覆盖、且明显领先时，才允许把整组记录锁定为该子类。
REAL_VALUE_SINGLE_VALUE_SH_MIN_RATIO = 0.95
REAL_VALUE_SINGLE_VALUE_SH_MIN_GAP = 0.20
REAL_VALUE_SINGLE_VALUE_SH_MIN_ENTITIES = 3
# Exact ObjectProperty names are useful discriminator evidence only after the
# value column is almost completely populated and its values resolve to a
# table that can represent the property's declared range.
REAL_VALUE_OBJECT_PROPERTY_MIN_COVERAGE = 0.95
REAL_VALUE_DATA_ATTR_NULL_FALLBACK_MIN_SCORE = 0.45      # 真实值增强 中数据属性重判返回 null 时允许回退 top1 的最低分

# ---------- R2RML ----------
R2RML_OP_FALLBACK_MIN_SCORE = 0.50      # R2RML OP fallback 允许落盘的最低综合分
R2RML_OP_FALLBACK_MIN_NAME_SCORE = 0.70 # R2RML OP fallback 允许落盘的最低名称分
R2RML_OP_FALLBACK_MIN_SIDE_SCORE = 0.3  # R2RML OP fallback 允许落盘的最低 domain/range 单侧分
R2RML_PK_OP_MIN_SCORE = 0.35            # 主键 FK 补 OP 允许落盘的最低综合分
R2RML_PK_OP_MIN_SIDE_SCORE = 0.3        # 主键 FK 补 OP 允许落盘的最低 domain/range 单侧分
R2RML_PK_OP_MIN_NAME_SCORE = 0.6        # 主键 FK 补 OP 允许落盘的最低名称分
R2RML_DP_REFINE_FILL_MIN_SCORE = 0.45   # 当前 DP 为空时，R2RML 自动补 DP 的最低综合分
R2RML_DP_REFINE_FILL_MIN_NAME_SCORE = 0.80 # 当前 DP 为空时，R2RML 自动补 DP 的最低名称分
R2RML_DP_REFINE_REPLACE_TOP_NAME = 0.75 # R2RML 替换当前 DP 时 top1 需要达到的名称分
R2RML_DP_REFINE_REPLACE_CUR_NAME = 0.35 # R2RML 替换当前 DP 时当前候选被视为较差的名称分上限
R2RML_DP_REFINE_REPLACE_NAME_GAP = 0.25 # R2RML 替换 DP 时要求 top1 与当前项的名称分差
R2RML_DP_REFINE_REPLACE_SCORE_GAP = 0.08 # R2RML 替换 DP 时要求 top1 与当前项的综合分差
R2RML_SH_INFER_SUBCLASS_MIN_SCORE = 0.7 # R2RML 为 SH 表推断 subclass 关系时的最低分


# ============================================================
#  Hyperparameter presets
#  说明：
#    1) main.py 父进程只负责按 HyperSelect 串行调度各挡位；
#    2) 真正跑 pipeline 的子进程通过环境变量 MAMG_HYPER_LEVEL 读取挡位；
#    3) 所有覆盖都收敛在 config.py，避免实验逻辑散落到业务代码里。
# ============================================================
if isHyper:
    if HyperSelect == "dp_weight":
        _weight_presets = {
            "w_0p9_0p1": (0.9, 0.1),
            "w_0p7_0p3": (0.7, 0.3),
            "w_0p5_0p5": (0.5, 0.5),
            "w_0p3_0p7": (0.3, 0.7),
        }
        DP_MAPPING_CANDIDATE_TEXT_WEIGHT, DP_MAPPING_CANDIDATE_DOMAIN_WEIGHT = _weight_presets.get(
            HyperLevel,
            _weight_presets["w_0p7_0p3"],
        )
    elif HyperSelect == "dp_confidence":
        _confidence_presets = {
            "loose":   (0.6, 0.15, 0.35, 0.05),
            "default": (0.7, 0.2,  0.4,  0.1),
            "strict":  (0.8, 0.25, 0.5,  0.15),
        }
        (
            DP_MAPPING_CONF_HIGH_TOP1,
            DP_MAPPING_CONF_HIGH_GAP,
            DP_MAPPING_CONF_MEDIUM_TOP1,
            DP_MAPPING_CONF_MEDIUM_GAP,
        ) = _confidence_presets.get(HyperLevel, _confidence_presets["default"])
    elif HyperSelect == "fk_completion_ind_threshold":
        _fk_ind_presets = {
            "0p6": 0.6,
            "0p8": 0.8,
            "0p9": 0.9,
            "0p95": 0.95,
            "1p0": 1.0,
        }
        FK_COMPLETION_IND_THRESHOLD = _fk_ind_presets.get(HyperLevel, _fk_ind_presets["0p95"])
