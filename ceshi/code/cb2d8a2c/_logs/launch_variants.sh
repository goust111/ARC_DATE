#!/usr/bin/env bash
set -u
cd /home/lixinyu/snap/new || exit 1
pids=()
names=()
(
  prompt=$(cat '/home/lixinyu/snap/new/ceshi/tem2/variant_prompt_cb2d8a2c_D2_alt0.txt')
  codex exec --skip-git-repo-check -c model='"gpt-5.4"' -c reasoning.effort='"high"' "$prompt" > '/home/lixinyu/snap/new/ceshi/code/cb2d8a2c/_logs/D2_alt0.log' 2>&1
  echo $? > '/home/lixinyu/snap/new/ceshi/code/cb2d8a2c/_logs/D2_alt0.status'
) &
pids+=("$!")
names+=('D2_alt0')
(
  prompt=$(cat '/home/lixinyu/snap/new/ceshi/tem2/variant_prompt_cb2d8a2c_D2_alt3.txt')
  codex exec --skip-git-repo-check -c model='"gpt-5.4"' -c reasoning.effort='"high"' "$prompt" > '/home/lixinyu/snap/new/ceshi/code/cb2d8a2c/_logs/D2_alt3.log' 2>&1
  echo $? > '/home/lixinyu/snap/new/ceshi/code/cb2d8a2c/_logs/D2_alt3.status'
) &
pids+=("$!")
names+=('D2_alt3')
(
  prompt=$(cat '/home/lixinyu/snap/new/ceshi/tem2/variant_prompt_cb2d8a2c_D3_alt0.txt')
  codex exec --skip-git-repo-check -c model='"gpt-5.4"' -c reasoning.effort='"high"' "$prompt" > '/home/lixinyu/snap/new/ceshi/code/cb2d8a2c/_logs/D3_alt0.log' 2>&1
  echo $? > '/home/lixinyu/snap/new/ceshi/code/cb2d8a2c/_logs/D3_alt0.status'
) &
pids+=("$!")
names+=('D3_alt0')
(
  prompt=$(cat '/home/lixinyu/snap/new/ceshi/tem2/variant_prompt_cb2d8a2c_D4_alt2.txt')
  codex exec --skip-git-repo-check -c model='"gpt-5.4"' -c reasoning.effort='"high"' "$prompt" > '/home/lixinyu/snap/new/ceshi/code/cb2d8a2c/_logs/D4_alt2.log' 2>&1
  echo $? > '/home/lixinyu/snap/new/ceshi/code/cb2d8a2c/_logs/D4_alt2.status'
) &
pids+=("$!")
names+=('D4_alt2')
(
  prompt=$(cat '/home/lixinyu/snap/new/ceshi/tem2/variant_prompt_cb2d8a2c_D4_alt3.txt')
  codex exec --skip-git-repo-check -c model='"gpt-5.4"' -c reasoning.effort='"high"' "$prompt" > '/home/lixinyu/snap/new/ceshi/code/cb2d8a2c/_logs/D4_alt3.log' 2>&1
  echo $? > '/home/lixinyu/snap/new/ceshi/code/cb2d8a2c/_logs/D4_alt3.status'
) &
pids+=("$!")
names+=('D4_alt3')
(
  prompt=$(cat '/home/lixinyu/snap/new/ceshi/tem2/variant_prompt_cb2d8a2c_D5_alt2.txt')
  codex exec --skip-git-repo-check -c model='"gpt-5.4"' -c reasoning.effort='"high"' "$prompt" > '/home/lixinyu/snap/new/ceshi/code/cb2d8a2c/_logs/D5_alt2.log' 2>&1
  echo $? > '/home/lixinyu/snap/new/ceshi/code/cb2d8a2c/_logs/D5_alt2.status'
) &
pids+=("$!")
names+=('D5_alt2')
(
  prompt=$(cat '/home/lixinyu/snap/new/ceshi/tem2/variant_prompt_cb2d8a2c_D5_alt3.txt')
  codex exec --skip-git-repo-check -c model='"gpt-5.4"' -c reasoning.effort='"high"' "$prompt" > '/home/lixinyu/snap/new/ceshi/code/cb2d8a2c/_logs/D5_alt3.log' 2>&1
  echo $? > '/home/lixinyu/snap/new/ceshi/code/cb2d8a2c/_logs/D5_alt3.status'
) &
pids+=("$!")
names+=('D5_alt3')
(
  prompt=$(cat '/home/lixinyu/snap/new/ceshi/tem2/variant_prompt_cb2d8a2c_D6_alt0.txt')
  codex exec --skip-git-repo-check -c model='"gpt-5.4"' -c reasoning.effort='"high"' "$prompt" > '/home/lixinyu/snap/new/ceshi/code/cb2d8a2c/_logs/D6_alt0.log' 2>&1
  echo $? > '/home/lixinyu/snap/new/ceshi/code/cb2d8a2c/_logs/D6_alt0.status'
) &
pids+=("$!")
names+=('D6_alt0')
(
  prompt=$(cat '/home/lixinyu/snap/new/ceshi/tem2/variant_prompt_cb2d8a2c_D8_alt1.txt')
  codex exec --skip-git-repo-check -c model='"gpt-5.4"' -c reasoning.effort='"high"' "$prompt" > '/home/lixinyu/snap/new/ceshi/code/cb2d8a2c/_logs/D8_alt1.log' 2>&1
  echo $? > '/home/lixinyu/snap/new/ceshi/code/cb2d8a2c/_logs/D8_alt1.status'
) &
pids+=("$!")
names+=('D8_alt1')
(
  prompt=$(cat '/home/lixinyu/snap/new/ceshi/tem2/variant_prompt_cb2d8a2c_D10_alt2.txt')
  codex exec --skip-git-repo-check -c model='"gpt-5.4"' -c reasoning.effort='"high"' "$prompt" > '/home/lixinyu/snap/new/ceshi/code/cb2d8a2c/_logs/D10_alt2.log' 2>&1
  echo $? > '/home/lixinyu/snap/new/ceshi/code/cb2d8a2c/_logs/D10_alt2.status'
) &
pids+=("$!")
names+=('D10_alt2')
(
  prompt=$(cat '/home/lixinyu/snap/new/ceshi/tem2/variant_prompt_cb2d8a2c_D13_alt2.txt')
  codex exec --skip-git-repo-check -c model='"gpt-5.4"' -c reasoning.effort='"high"' "$prompt" > '/home/lixinyu/snap/new/ceshi/code/cb2d8a2c/_logs/D13_alt2.log' 2>&1
  echo $? > '/home/lixinyu/snap/new/ceshi/code/cb2d8a2c/_logs/D13_alt2.status'
) &
pids+=("$!")
names+=('D13_alt2')
(
  prompt=$(cat '/home/lixinyu/snap/new/ceshi/tem2/variant_prompt_cb2d8a2c_D15_alt1.txt')
  codex exec --skip-git-repo-check -c model='"gpt-5.4"' -c reasoning.effort='"high"' "$prompt" > '/home/lixinyu/snap/new/ceshi/code/cb2d8a2c/_logs/D15_alt1.log' 2>&1
  echo $? > '/home/lixinyu/snap/new/ceshi/code/cb2d8a2c/_logs/D15_alt1.status'
) &
pids+=("$!")
names+=('D15_alt1')
(
  prompt=$(cat '/home/lixinyu/snap/new/ceshi/tem2/variant_prompt_cb2d8a2c_D16_alt0.txt')
  codex exec --skip-git-repo-check -c model='"gpt-5.4"' -c reasoning.effort='"high"' "$prompt" > '/home/lixinyu/snap/new/ceshi/code/cb2d8a2c/_logs/D16_alt0.log' 2>&1
  echo $? > '/home/lixinyu/snap/new/ceshi/code/cb2d8a2c/_logs/D16_alt0.status'
) &
pids+=("$!")
names+=('D16_alt0')
(
  prompt=$(cat '/home/lixinyu/snap/new/ceshi/tem2/variant_prompt_cb2d8a2c_D17_alt0.txt')
  codex exec --skip-git-repo-check -c model='"gpt-5.4"' -c reasoning.effort='"high"' "$prompt" > '/home/lixinyu/snap/new/ceshi/code/cb2d8a2c/_logs/D17_alt0.log' 2>&1
  echo $? > '/home/lixinyu/snap/new/ceshi/code/cb2d8a2c/_logs/D17_alt0.status'
) &
pids+=("$!")
names+=('D17_alt0')
(
  prompt=$(cat '/home/lixinyu/snap/new/ceshi/tem2/variant_prompt_cb2d8a2c_D18_alt1.txt')
  codex exec --skip-git-repo-check -c model='"gpt-5.4"' -c reasoning.effort='"high"' "$prompt" > '/home/lixinyu/snap/new/ceshi/code/cb2d8a2c/_logs/D18_alt1.log' 2>&1
  echo $? > '/home/lixinyu/snap/new/ceshi/code/cb2d8a2c/_logs/D18_alt1.status'
) &
pids+=("$!")
names+=('D18_alt1')
(
  prompt=$(cat '/home/lixinyu/snap/new/ceshi/tem2/variant_prompt_cb2d8a2c_D18_alt2.txt')
  codex exec --skip-git-repo-check -c model='"gpt-5.4"' -c reasoning.effort='"high"' "$prompt" > '/home/lixinyu/snap/new/ceshi/code/cb2d8a2c/_logs/D18_alt2.log' 2>&1
  echo $? > '/home/lixinyu/snap/new/ceshi/code/cb2d8a2c/_logs/D18_alt2.status'
) &
pids+=("$!")
names+=('D18_alt2')
(
  prompt=$(cat '/home/lixinyu/snap/new/ceshi/tem2/variant_prompt_cb2d8a2c_D19_alt1.txt')
  codex exec --skip-git-repo-check -c model='"gpt-5.4"' -c reasoning.effort='"high"' "$prompt" > '/home/lixinyu/snap/new/ceshi/code/cb2d8a2c/_logs/D19_alt1.log' 2>&1
  echo $? > '/home/lixinyu/snap/new/ceshi/code/cb2d8a2c/_logs/D19_alt1.status'
) &
pids+=("$!")
names+=('D19_alt1')
(
  prompt=$(cat '/home/lixinyu/snap/new/ceshi/tem2/variant_prompt_cb2d8a2c_D20_alt0.txt')
  codex exec --skip-git-repo-check -c model='"gpt-5.4"' -c reasoning.effort='"high"' "$prompt" > '/home/lixinyu/snap/new/ceshi/code/cb2d8a2c/_logs/D20_alt0.log' 2>&1
  echo $? > '/home/lixinyu/snap/new/ceshi/code/cb2d8a2c/_logs/D20_alt0.status'
) &
pids+=("$!")
names+=('D20_alt0')
(
  prompt=$(cat '/home/lixinyu/snap/new/ceshi/tem2/variant_prompt_cb2d8a2c_D20_alt1.txt')
  codex exec --skip-git-repo-check -c model='"gpt-5.4"' -c reasoning.effort='"high"' "$prompt" > '/home/lixinyu/snap/new/ceshi/code/cb2d8a2c/_logs/D20_alt1.log' 2>&1
  echo $? > '/home/lixinyu/snap/new/ceshi/code/cb2d8a2c/_logs/D20_alt1.status'
) &
pids+=("$!")
names+=('D20_alt1')
for i in "${!pids[@]}"; do
  pid=${pids[$i]}
  name=${names[$i]}
  wait "$pid"
  rc=$?
  printf "%s %s
" "$name" "$rc"
done
