#!/usr/bin/env bash
set -euo pipefail

# ====== 기본 설정 ======
CONTAINER_NAME="hailo8_ai_sw_suite_2025-10_container"
IMAGE_NAME="hailo8_ai_sw_suite_2025-10:1"
SHARED_DIR="shared_with_docker"   # 호스트: ./shared_with_docker -> 컨테이너: /local/shared_with_docker

# ====== 옵션 파싱 ======
MODE="new"         # new | resume | override
USE_GPU="yes"       # no  | yes

while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume)   MODE="resume"; shift ;;
    --override) MODE="override"; shift ;;
    --gpu)      USE_GPU="yes"; shift ;;
    -h|--help)
      cat <<EOF
Usage:
  $0 [--resume | --override] [--gpu]

Options:
  --resume     기존 컨테이너 실행/접속
  --override   기존 컨테이너 삭제 후 새로 생성
  --gpu        GPU 활성화(--gpus all). NVML 미스매치 있으면 빼세요.
EOF
      exit 0
      ;;
    *)
      echo "Unknown option: $1"; exit 1 ;;
  esac
done

# ====== 공용 함수 ======
exists_container() {
  docker ps -a --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME"
}

start_shell() {
  docker exec -ti "$CONTAINER_NAME" /bin/bash
}

create_shared() {
  mkdir -p "$SHARED_DIR"
  chmod 777 "$SHARED_DIR" || true
}

run_new() {
  create_shared

  DOCKER_ARGS=(
    # --privileged
    --net=host
    --ipc=host
    --name "$CONTAINER_NAME"
    -v "/dev:/dev"
    -v "/lib/firmware:/lib/firmware"
    -v "/lib/modules:/lib/modules"
    -v "/lib/udev/rules.d:/lib/udev/rules.d"
    -v "/usr/src:/usr/src"
    -v "$(pwd)/${SHARED_DIR}:/local/${SHARED_DIR}:rw"
    -v "/etc/timezone:/etc/timezone:ro"
    -v "/etc/localtime:/etc/localtime:ro"
    -d -it --rm
  )

  # DKMS 디렉토리가 있으면 마운트(커널 모듈 빌드 환경 필요시)
  if [[ -d "/var/lib/dkms" ]]; then
    DOCKER_ARGS+=(-v "/var/lib/dkms:/var/lib/dkms")
  fi

  # GPU 옵션 (기본 off)
  if [[ "$USE_GPU" == "yes" ]]; then
    DOCKER_ARGS+=(--gpus "device=0")
  fi

  echo ">>> docker run ${DOCKER_ARGS[*]} -ti $IMAGE_NAME"
  docker run "${DOCKER_ARGS[@]}" -ti "$IMAGE_NAME"
}

# ====== Docker 이미지 확인 및 로드 ======
check_and_load_image() {
  if ! docker images --format '{{.Repository}}:{{.Tag}}' | grep -Fxq "$IMAGE_NAME"; then
    echo "Docker 이미지 '$IMAGE_NAME'를 찾을 수 없습니다."
    
    if [[ -f "hailo8_ai_sw_suite_2025-10.tar.gz" ]]; then
      echo "tar.gz 파일에서 이미지를 로드합니다..."
      docker load -i hailo8_ai_sw_suite_2025-10.tar.gz
      
      # 로드 후 다시 확인
      if ! docker images --format '{{.Repository}}:{{.Tag}}' | grep -Fxq "$IMAGE_NAME"; then
        echo "에러: tar.gz 파일에서 이미지를 로드했지만 '$IMAGE_NAME'를 찾을 수 없습니다."
        exit 1
      fi
      echo "Docker 이미지를 성공적으로 로드했습니다."
    else
      echo "에러: Docker 이미지 '$IMAGE_NAME'와 tar.gz 파일 'hailo8_ai_sw_suite_2025-10.tar.gz' 모두 찾을 수 없습니다."
      echo "다음 명령으로 이미지를 수동으로 로드하세요:"
      echo "  docker load -i hailo8_ai_sw_suite_2025-10.tar.gz"
      exit 1
    fi
  fi
}

# ====== 모드별 실행 ======
case "$MODE" in
  new)
    if exists_container; then
      echo "컨테이너가 이미 존재합니다. --resume 또는 --override 사용하세요."
      exit 1
    fi
    # 이미지 존재 여부 확인 및 필요시 로드
    check_and_load_image
    run_new
    ;;

  override)
    if exists_container; then
      echo "기존 컨테이너 삭제..."
      docker rm -f "$CONTAINER_NAME" >/dev/null || true
    fi
    # 이미지 존재 여부 확인 및 필요시 로드
    check_and_load_image
    run_new
    ;;

  resume)
    if ! exists_container; then
      echo "컨테이너가 없습니다. 먼저 생성하세요: $0  또는  $0 --override"
      exit 1
    fi
    docker start "$CONTAINER_NAME" >/dev/null
    start_shell
    ;;

  *)
    echo "알 수 없는 모드: $MODE"; exit 1 ;;
esac