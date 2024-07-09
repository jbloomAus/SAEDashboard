# docker build --target development -t saedashboard-cuda --file Dockerfile .
# docker run --entrypoint /bin/bash -it saedashboard-cuda

ARG APP_NAME=sae_dashboard
ARG APP_PATH=/opt/$APP_NAME
ARG PYTHON_VERSION=3.12.2
ARG POETRY_VERSION=1.8.3

FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel as staging
ARG APP_NAME
ARG APP_PATH
ARG POETRY_VERSION

ENV \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1
ENV \
    POETRY_VERSION=$POETRY_VERSION \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1

RUN apt-get update && apt-get -y install curl
RUN curl -sSL https://install.python-poetry.org | python
ENV PATH="$POETRY_HOME/bin:$PATH"

WORKDIR $APP_PATH
COPY ./poetry.lock ./pyproject.toml ./README.md ./
COPY ./$APP_NAME ./$APP_NAME

FROM staging as development
ARG APP_NAME
ARG APP_PATH

WORKDIR $APP_PATH
RUN poetry install

ENTRYPOINT ["/bin/bash"]
CMD ["poetry", "shell"]