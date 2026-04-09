import os

SECRET_KEY = "football_analytics_superset_secret_key"

SQLALCHEMY_DATABASE_URI = os.getenv(
    "SQLALCHEMY_DATABASE_URI",
    "postgresql+psycopg2://superset:superset123@superset-db:5432/superset",
)

ROW_LIMIT = 5000
SQLLAB_CTAS_NO_LIMIT = True
WTF_CSRF_ENABLED = True

FEATURE_FLAGS = {
    "ENABLE_TEMPLATE_PROCESSING": True,
}