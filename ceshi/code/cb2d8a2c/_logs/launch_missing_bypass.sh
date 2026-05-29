#!/usr/bin/env bash
set -u
cd /home/lixinyu/snap/new || exit 1
pids=()
names=()
(
  prompt=$(cat '/home/lixinyu/snap/new/ceshi/tem2/variant_prompt_cb2d8a2c_D2_alt0.txt')
  codex exec --skip-git-repo-check --dangerously-bypass-approvals-and-sandbox -C /home/lixinyu/snap/new -c model='"gpt-5.4"' -c reasoning.effort='"high"' "$prompt" > '/home/lixinyu/snap/new/ceshi/code/cb2d8a2c/_logs/D2_alt0_solo.log' 2>&1
  echo $? > '/home/lixinyu/snap/new/ceshi/code/cb2d8a2c/_logs/D2_alt0_solo.status'
) &
pids+=("$!")
names+=('D2_alt0')
(
  prompt=$(cat '/home/lixinyu/snap/new/ceshi/tem2/variant_prompt_cb2d8a2c_D10_alt2.txt')
  codex exec --skip-git-repo-check --dangerously-bypass-approvals-and-sandbox -C /home/lixinyu/snap/new -c model='"gpt-5.4"' -c reasoning.effort='"high"' "$prompt" > '/home/lixinyu/snap/new/ceshi/code/cb2d8a2c/_logs/D10_alt2_solo.log' 2>&1
  echo $? > '/home/lixinyu/snap/new/ceshi/code/cb2d8a2c/_logs/D10_alt2_solo.status'
) &
pids+=("$!")
names+=('D10_alt2')
(
  prompt=$(cat '/home/lixinyu/snap/new/ceshi/tem2/variant_prompt_cb2d8a2c_D13_alt2.txt')
  codex exec --skip-git-repo-check --dangerously-bypass-approvals-and-sandbox -C /home/lixinyu/snap/new -c model='"gpt-5.4"' -c reasoning.effort='"high"' "$prompt" > '/home/lixinyu/snap/new/ceshi/code/cb2d8a2c/_logs/D13_alt2_solo.log' 2>&1
  echo $? > '/home/lixinyu/snap/new/ceshi/code/cb2d8a2c/_logs/D13_alt2_solo.status'
) &
pids+=("$!")
names+=('D13_alt2')
(
  prompt=$(cat '/home/lixinyu/snap/new/ceshi/tem2/variant_prompt_cb2d8a2c_D16_alt0.txt')
  codex exec --skip-git-repo-check --dangerously-bypass-approvals-and-sandbox -C /home/lixinyu/snap/new -c model='"gpt-5.4"' -c reasoning.effort='"high"' "$prompt" > '/home/lixinyu/snap/new/ceshi/code/cb2d8a2c/_logs/D16_alt0_solo.log' 2>&1
  echo $? > '/home/lixinyu/snap/new/ceshi/code/cb2d8a2c/_logs/D16_alt0_solo.status'
) &
pids+=("$!")
names+=('D16_alt0')
(
  prompt=$(cat '/home/lixinyu/snap/new/ceshi/tem2/variant_prompt_cb2d8a2c_D17_alt0.txt')
  codex exec --skip-git-repo-check --dangerously-bypass-approvals-and-sandbox -C /home/lixinyu/snap/new -c model='"gpt-5.4"' -c reasoning.effort='"high"' "$prompt" > '/home/lixinyu/snap/new/ceshi/code/cb2d8a2c/_logs/D17_alt0_solo.log' 2>&1
  echo $? > '/home/lixinyu/snap/new/ceshi/code/cb2d8a2c/_logs/D17_alt0_solo.status'
) &
pids+=("$!")
names+=('D17_alt0')
for i in "${!pids[@]}"; do
  pid=${pids[$i]}
  name=${names[$i]}
  wait "$pid"
  rc=$?
  printf "%s %s
" "$name" "$rc"
done
