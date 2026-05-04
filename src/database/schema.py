"""
Definições de schema do banco de dados SQLite.

Contém as instruções DDL para criação de tabelas e índices
do sistema de persistência de avaliações multiagente.
"""

_CREATE_EVALUATION_RUNS = """
CREATE TABLE IF NOT EXISTS evaluation_runs (
    run_id          TEXT PRIMARY KEY,
    llm_model       TEXT NOT NULL,
    llm_temperature REAL NOT NULL DEFAULT 0.7,
    profiles_used   TEXT NOT NULL,
    product_count   INTEGER,
    notes           TEXT,
    created_at      TEXT NOT NULL
)
"""

_CREATE_PRODUCTS = """
CREATE TABLE IF NOT EXISTS products (
    product_id     TEXT PRIMARY KEY,
    title          TEXT NOT NULL,
    brand          TEXT,
    category       TEXT,
    current_price  REAL,
    original_price REAL,
    discount_pct   REAL,
    rating         REAL,
    review_count   INTEGER
)
"""

_CREATE_PRODUCT_EVALUATIONS = """
CREATE TABLE IF NOT EXISTS product_evaluations (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id                TEXT    NOT NULL REFERENCES evaluation_runs(run_id),
    product_id            TEXT    NOT NULL REFERENCES products(product_id),
    mean_score            REAL,
    min_score             REAL,
    max_score             REAL,
    coverage_score        REAL,
    risk_score            REAL,
    consensus_level       REAL,
    overall_appeal        REAL,
    conversion_potential  REAL,
    improvement_potential REAL,
    profiles_analyzed     INTEGER,
    generated_at          TEXT,
    UNIQUE (run_id, product_id)
)
"""

_CREATE_PROFILE_EVALUATIONS = """
CREATE TABLE IF NOT EXISTS profile_evaluations (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    product_evaluation_id INTEGER NOT NULL REFERENCES product_evaluations(id),
    profile               TEXT    NOT NULL,
    score                 REAL,
    purchase_intention    REAL,
    status                TEXT
)
"""

_CREATE_PROFILE_CONCERNS = """
CREATE TABLE IF NOT EXISTS profile_concerns (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_evaluation_id INTEGER NOT NULL REFERENCES profile_evaluations(id),
    position              INTEGER NOT NULL,
    concern               TEXT    NOT NULL
)
"""

_CREATE_PROFILE_STRENGTHS = """
CREATE TABLE IF NOT EXISTS profile_strengths (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_evaluation_id INTEGER NOT NULL REFERENCES profile_evaluations(id),
    position              INTEGER NOT NULL,
    strength              TEXT    NOT NULL
)
"""

_CREATE_CONSENSUS_ITEMS = """
CREATE TABLE IF NOT EXISTS consensus_items (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    product_evaluation_id INTEGER NOT NULL REFERENCES product_evaluations(id),
    type                  TEXT    NOT NULL CHECK (type IN ('strength', 'weakness', 'disagreement_area')),
    position              INTEGER NOT NULL,
    value                 TEXT    NOT NULL
)
"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_pe_run     ON product_evaluations(run_id)",
    "CREATE INDEX IF NOT EXISTS idx_pe_product ON product_evaluations(product_id)",
    "CREATE INDEX IF NOT EXISTS idx_pfe_pe     ON profile_evaluations(product_evaluation_id)",
    "CREATE INDEX IF NOT EXISTS idx_ci_pe      ON consensus_items(product_evaluation_id)",
    "CREATE INDEX IF NOT EXISTS idx_pc_pfe     ON profile_concerns(profile_evaluation_id)",
    "CREATE INDEX IF NOT EXISTS idx_ps_pfe     ON profile_strengths(profile_evaluation_id)",
]

ALL_TABLES: list[str] = [
    _CREATE_EVALUATION_RUNS,
    _CREATE_PRODUCTS,
    _CREATE_PRODUCT_EVALUATIONS,
    _CREATE_PROFILE_EVALUATIONS,
    _CREATE_PROFILE_CONCERNS,
    _CREATE_PROFILE_STRENGTHS,
    _CREATE_CONSENSUS_ITEMS,
]

ALL_INDEXES: list[str] = _CREATE_INDEXES
