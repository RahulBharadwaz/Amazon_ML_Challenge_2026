#!/usr/bin/env bash
# Ephemeral EC2 work session for the ER project (ap-south-1).
#
#   infra/ec2_session.sh up [instance-type]   launch + swap + Python deps + sync dataset/$ER_DATASETS (default: train)
#   infra/ec2_session.sh push <files...>      copy local files into ~/er on the instance
#   infra/ec2_session.sh run "<command>"      run a command in ~/er (streams output)
#   infra/ec2_session.sh pull <remote> <dir>  copy ~/er/<remote> back to a local directory
#   infra/ec2_session.sh down                 terminate and verify 'terminated'
#   infra/ec2_session.sh status
#
# Safety: shutdown behaviour is 'terminate' and the instance schedules its own
# shutdown after DEADMAN_MIN minutes, so a forgotten session cannot run up credits.
# S3 access comes from the er-ml-ec2-profile instance role (infra/er-ml-iam.yaml);
# no user credentials ever reach the instance.
set -uo pipefail
export AWS_REGION=ap-south-1
export MSYS_NO_PATHCONV=1   # Git Bash on Windows: don't rewrite /dev/xvda etc. into Windows paths

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_DIR="${ER_SESSION_DIR:-$PROJ/.ec2_session}"
KEY="${ER_KEY:-$PROJ/.secrets/ml-challenge-key.pem}"
KH="$STATE_DIR/known_hosts"
SG_NAME=ml-challenge-sg
KEY_NAME=ml-challenge-key
PROFILE=er-ml-ec2-profile
BUCKET=s3://hackathon-tensors-2026
DEADMAN_MIN="${DEADMAN_MIN:-180}"
mkdir -p "$STATE_DIR"

log(){ echo "[$(date +%H:%M:%S)] $*"; }
iid(){ cat "$STATE_DIR/instance_id" 2>/dev/null; }
ip(){ cat "$STATE_DIR/ip" 2>/dev/null; }
ssh_(){ ssh -i "$KEY" -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" \
          -o ConnectTimeout=10 -o ServerAliveInterval=30 ec2-user@"$(ip)" "$@"; }

cmd_up(){
  local type="${1:-m6i.xlarge}"
  [ -n "$(iid)" ] && { log "session already active: $(iid)"; return 1; }
  local ami sg
  ami=$(aws ec2 describe-images --owners amazon \
        --filters "Name=name,Values=al2023-ami-2023*-kernel-*-x86_64" "Name=state,Values=available" \
        --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text) || return 1
  sg=$(aws ec2 describe-security-groups --filters Name=group-name,Values=$SG_NAME \
       --query 'SecurityGroups[0].GroupId' --output text) || return 1
  log "Launching $type ($ami) with profile $PROFILE, dead-man ${DEADMAN_MIN} min..."
  local id
  id=$(aws ec2 run-instances --image-id "$ami" --instance-type "$type" --key-name $KEY_NAME \
    --security-group-ids "$sg" --iam-instance-profile Name=$PROFILE \
    --instance-initiated-shutdown-behavior terminate \
    --metadata-options HttpTokens=required,HttpEndpoint=enabled \
    --block-device-mappings 'DeviceName=/dev/xvda,Ebs={VolumeSize=40,VolumeType=gp3,Encrypted=true,DeleteOnTermination=true}' \
    --user-data "$(printf '#!/bin/bash\nshutdown -h +%s "ER dead-man switch"\n' "$DEADMAN_MIN")" \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=er-ml-session},{Key=Project,Value=amazon-ml-2026}]' \
    --query 'Instances[0].InstanceId' --output text) || return 1
  echo "$id" > "$STATE_DIR/instance_id"
  log "Instance $id; waiting for running..."
  aws ec2 wait instance-running --instance-ids "$id" || return 1
  aws ec2 describe-instances --instance-ids "$id" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text > "$STATE_DIR/ip"
  log "Running at $(ip); waiting for SSH..."
  for _ in $(seq 1 30); do ssh_ true 2>/dev/null && break; sleep 6; done
  ssh_ true || { log "SSH never came up"; return 1; }
  log "Configuring swap, Python 3.11 and ML deps..."
  ssh_ 'set -e
    sudo fallocate -l 8G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile >/dev/null && sudo swapon /swapfile
    sudo dnf install -y -q python3.11 python3.11-pip libgomp >/dev/null
    python3.11 -m pip install -q --user rapidfuzz pandas pyarrow lightgbm numpy psutil
    free -h | head -2
    python3.11 -c "import lightgbm,rapidfuzz,pyarrow,pandas; print(\"lightgbm\",lightgbm.__version__,\"| rapidfuzz\",rapidfuzz.__version__,\"| pyarrow\",pyarrow.__version__,\"| pandas\",pandas.__version__)"' || return 1
  local ds
  for ds in ${ER_DATASETS:-train}; do
    log "Syncing dataset/$ds from S3 (instance role)..."
    ssh_ "set -e; mkdir -p ~/er/dataset/$ds; t=\$(date +%s)
      aws s3 sync $BUCKET/dataset/$ds/ ~/er/dataset/$ds/ --only-show-errors --region $AWS_REGION
      du -sh ~/er/dataset/$ds; echo \"sync took \$(( \$(date +%s)-t ))s\"" || return 1
  done
  log "Session ready: $(iid) @ $(ip)"
}

cmd_push(){ scp -q -i "$KEY" -o UserKnownHostsFile="$KH" "$@" ec2-user@"$(ip)":~/er/ && log "pushed: $*"; }
cmd_run(){ ssh_ "cd ~/er && export PYTHONIOENCODING=utf-8 && $1"; }
cmd_pull(){ mkdir -p "$2"; scp -q -r -i "$KEY" -o UserKnownHostsFile="$KH" "ec2-user@$(ip):~/er/$1" "$2/" && log "pulled $1 -> $2"; }

cmd_down(){
  local id; id=$(iid)
  [ -z "$id" ] && { log "no active session"; return 0; }
  log "Terminating $id..."
  aws ec2 terminate-instances --instance-ids "$id" --query 'TerminatingInstances[0].CurrentState.Name' --output text
  aws ec2 wait instance-terminated --instance-ids "$id" || { log "WARNING: wait failed, check $id"; return 1; }
  log "VERIFIED STATE: $(aws ec2 describe-instances --instance-ids "$id" --query 'Reservations[0].Instances[0].State.Name' --output text)"
  rm -f "$STATE_DIR/instance_id" "$STATE_DIR/ip" "$KH"
}

cmd_status(){
  local id; id=$(iid)
  [ -z "$id" ] && { log "no active session"; return 0; }
  aws ec2 describe-instances --instance-ids "$id" \
    --query 'Reservations[0].Instances[0].[InstanceId,InstanceType,State.Name,PublicIpAddress,LaunchTime]' --output text
}

case "${1:-}" in
  up) shift; cmd_up "$@" ;;
  push) shift; cmd_push "$@" ;;
  run) shift; cmd_run "$1" ;;
  pull) shift; cmd_pull "$1" "$2" ;;
  down) cmd_down ;;
  status) cmd_status ;;
  *) sed -n '2,15p' "$0"; exit 2 ;;
esac
