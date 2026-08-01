from pathlib import Path
import environ

env = environ.Env(DEBUG=(bool, False))
BASE_DIR = Path(__file__).resolve().parent.parent
environ.Env.read_env(BASE_DIR / ".env")

SECRET_KEY = env("SECRET")
DEBUG = True

ALLOWED_HOSTS = ['casa.leoandjen.com', '192.168.1.2', '127.0.0.1', 'emo-server', 'localhost']

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'core',
    'wemo',
    'ai_lab_core',
    'ai_lab_chatbot',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'casa.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'casa.wsgi.application'

# You are running only one DB, on the server, not locally
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'django_project2',
        'USER': 'django_user',
        'PASSWORD': env("CASA_DB_PASS"),
        'HOST': '127.0.0.1',
        'PORT': '5432',
    },
    'ai_lab': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'ai_lab',
        'USER': 'ai_lab_user',
        'PASSWORD': env("AI_LAB_DB_PASS"),
        'HOST': '127.0.0.1',
        'PORT': '5432',
    },
}

DATABASE_ROUTERS = ['ai_lab_core.routers.AiLabRouter']

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'America/New_York'
USE_I18N = True
USE_TZ = True

STATIC_URL = 'static/'
STATIC_ROOT = '/home/leo/workspace/casa/static'

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Ollama (AI Lab / Mycroft). Django and Ollama run on the same box (emo-server),
# so localhost is the default; override in .env if Django runs elsewhere.
OLLAMA_HOST = env("OLLAMA_HOST", default="http://localhost:11434")
OLLAMA_CHAT_MODEL = env("OLLAMA_CHAT_MODEL", default="llama3.1:8b")

# How many trailing messages of a conversation are sent to Ollama as context
# (the sliding window). The full transcript is always stored and displayed.
MYCROFT_HISTORY_WINDOW = env.int("MYCROFT_HISTORY_WINDOW", default=20)

# Ollama context window (tokens). Drives both the actual chat call (passed as
# num_ctx in the request options) and the UI's "N / <budget>" percentage, so the
# figure the user sees matches what Ollama actually allots.
MYCROFT_NUM_CTX = env.int("MYCROFT_NUM_CTX", default=8192)

# Embedding model for semantic memory (Phase 3). Messages and Knowledge entries
# are embedded with this and stored as pgvector columns for similarity search.
OLLAMA_EMBED_MODEL = env("OLLAMA_EMBED_MODEL", default="nomic-embed-text")

# Semantic-recall knobs (tunable via .env without a code change). Each new user
# turn pulls up to this many similar past messages (the user's own, from other
# conversations) and curated Knowledge entries into the system prompt.
MYCROFT_RETRIEVAL_MESSAGES = env.int("MYCROFT_RETRIEVAL_MESSAGES", default=5)
MYCROFT_RETRIEVAL_KNOWLEDGE = env.int("MYCROFT_RETRIEVAL_KNOWLEDGE", default=3)
# Cosine-distance ceiling; matches farther than this are treated as irrelevant
# and dropped, so weak matches don't pollute the prompt.
MYCROFT_RETRIEVAL_MAX_DISTANCE = env.float("MYCROFT_RETRIEVAL_MAX_DISTANCE", default=0.55)

LOGIN_REDIRECT_URL = '/'
LOGOUT_REDIRECT_URL = '/login/'
LOGIN_URL = 'login'

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,

    'formatters': {
        'verbose': {
            'format': '[{asctime}] {levelname} {name}: {message}',
            'style': '{',
            'datefmt': '%Y-%m-%d %H:%M:%S',
        },
    },

    'handlers': {
        'file': {
            'level': 'DEBUG',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': '/var/log/django/debug.log',
            'formatter': 'verbose',
            'maxBytes': 50 * 1024 * 1024,  # 50MB
            'backupCount': 5,
        },
        'away_mode_file': {
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': '/var/log/django/away_mode.log',
            'formatter': 'verbose',
            'maxBytes': 10 * 1024 * 1024,  # 10MB
            'backupCount': 3,
        },
    },

    'loggers': {
        'django': {
            'handlers': ['file'],
            'level': 'INFO',
            'propagate': True,
        },
        'core.views': {
            'handlers': ['file'],
            'level': 'DEBUG',
            'propagate': False,
        },
        'ai_lab_chatbot.views': {
            'handlers': ['file'],
            'level': 'INFO',
            'propagate': False,
        },
        'away_mode': {
            'handlers': ['away_mode_file'],
            'level': 'INFO',
            'propagate': False,
        },
    },
}
