#!/bin/bash -e

#SBATCH --job-name="argument-quality"
#SBATCH --array=1-4
#SBATCH --output="/mnt/ceph/storage/data-tmp/current/kipu5728/log-%x-%j-%a.txt"
#SBATCH --error="/mnt/ceph/storage/data-tmp/current/kipu5728/log-%x-%j-%a.err.txt"
#SBATCH --gres=gpu:ampere
#SBATCH --mem=64g

CONTAINER_NAME="argument-quality"
IMAGE_SQSH="${CONTAINER_NAME}.sqsh"
IMAGE="docker://registry.webis.de#code-lib/public-images/llms-as-argument-quality-rater:latest"

echo "Check container"

mapfile -t AVAILABLE_CONTAINER < <( enroot list )

if [[ ! " ${AVAILABLE_CONTAINER[*]} " =~ ${CONTAINER_NAME} ]]; then
  if [ ! -f "$IMAGE_SQSH" ]; then
    echo "Can't find image \"${IMAGE_SQSH}\"..."
    enroot import -o ${IMAGE_SQSH} "${IMAGE}"
  fi

  echo "Create container \"${CONTAINER_NAME}\"..."
  enroot create --name ${CONTAINER_NAME} ${IMAGE_SQSH}
fi

enroot start -w -r -m /mnt/ceph/storage/data-tmp/current/kipu5728/argument-quality/src:/app/src ${CONTAINER_NAME}

