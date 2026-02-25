$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$OutDir = Join-Path $RepoRoot "out"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$ConsoleLog = Join-Path $OutDir "train_console.log"
$PidFile = Join-Path $OutDir "train.pid"

$PythonExe = if ($env:PYTHON_EXECUTABLE -and $env:PYTHON_EXECUTABLE.Trim().Length -gt 0) {
  $env:PYTHON_EXECUTABLE
} else {
  "python"
}

$TrainArgs = @(
  "-m", "python_ai.lettergen.train",
  "--infinite",
  "--resume", "auto",
  "--save-every", "1",
  "--log-file", "out/train.log"
)

$QuotedTrain = ($TrainArgs | ForEach-Object { $_ }) -join " "
$CmdLine = "/c cd /d `"$RepoRoot`" && `"$PythonExe`" $QuotedTrain >> `"$ConsoleLog`" 2>&1"

$Proc = Start-Process -FilePath "cmd.exe" `
  -ArgumentList $CmdLine `
  -WorkingDirectory $RepoRoot `
  -WindowStyle Hidden `
  -PassThru

$Proc.Id | Set-Content -Path $PidFile -Encoding ascii

Write-Output "Started python_ai.lettergen.train in background."
Write-Output "PID: $($Proc.Id)"
Write-Output "Repo: $RepoRoot"
Write-Output "Train log: $(Join-Path $OutDir 'train.log')"
Write-Output "Console log: $ConsoleLog"
Write-Output "PID file: $PidFile"

