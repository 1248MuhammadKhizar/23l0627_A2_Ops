$runs = @(
  @{n="r1_base";        LR="1e-3"; AUG="1"; SIZE="128"; UNF="0";  BB="mobilenetv2"},
  @{n="r2_lr_1e-4";     LR="1e-4"; AUG="1"; SIZE="128"; UNF="0";  BB="mobilenetv2"},
  @{n="r3_no_aug";      LR="1e-3"; AUG="0"; SIZE="128"; UNF="0";  BB="mobilenetv2"},
  @{n="r4_unfreeze10";  LR="1e-3"; AUG="1"; SIZE="128"; UNF="10"; BB="mobilenetv2"},
  @{n="r5_size96";      LR="1e-3"; AUG="1"; SIZE="96";  UNF="0";  BB="mobilenetv2"},
  @{n="r6_efficientnet";LR="1e-3"; AUG="1"; SIZE="128"; UNF="0";  BB="efficientnetb0"}
)
foreach ($r in $runs) {
  $env:RUN_NAME = $r.n; $env:LR = $r.LR; $env:AUG = $r.AUG
  $env:IMG_SIZE = $r.SIZE; $env:UNF = $r.UNF; $env:BB = $r.BB
  Write-Host "=== $($r.n)"
  python scripts\train_run.py 2>&1 | Select-String "done|Error|Traceback"
}
