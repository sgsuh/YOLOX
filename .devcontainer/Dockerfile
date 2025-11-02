FROM nvcr.io/nvidia/pytorch:20.12-py3

RUN apt update && \
    apt install -y sudo

ARG USER_NAME
ARG USER_UID=1000
ARG USER_GID=${USER_UID}

RUN groupadd --gid ${USER_GID} ${USER_NAME} \
  && useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USER_NAME} \
  && echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USER_NAME} \
  && echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers \
  && usermod -a -G video ${USER_NAME} \
  && chmod 0440 /etc/sudoers.d/${USER_NAME}
USER ${USER_NAME}

ENV SHELL=/bin/bash
SHELL ["/bin/bash", "-c"]