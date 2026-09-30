[CmdletBinding()]
param(
    [string]$Project,
    [switch]$Interactive,
    [int]$Threads = 8,
    [switch]$Fresh,
    [switch]$Clean,
    [switch]$SkipBuild,
    [switch]$DryRun,
    [switch]$SelfTest
)

$ErrorActionPreference = 'Continue'

$Root   = Split-Path -Parent $PSScriptRoot
$Image  = 'debug907:submission'
$Volume = 'debug907-out'
$Out    = Join-Path $Root 'out'
$step   = 0

$dockerCmd = Get-Command docker -CommandType Application -ErrorAction SilentlyContinue |
             Select-Object -First 1
if (-not $dockerCmd) {
    Write-Host "docker.exe not found on PATH. Install Docker Desktop first." -ForegroundColor Red
    exit 1
}
$DockerExe = $dockerCmd.Source

function Write-Step([string]$title) {
    $script:step++
    Write-Host ""
    Write-Host ("== {0}. {1} ==" -f $script:step, $title) -ForegroundColor Cyan
}

function Stop-Pipeline([string]$message) {
    Write-Host ""
    Write-Host "FAILED at step ${script:step}: $message" -ForegroundColor Red
    exit 1
}

$InfoTimeout = 15
function Test-DockerDaemon {
    $job = Start-Job -ScriptBlock { param($exe) & $exe info *> $null; $LASTEXITCODE } -ArgumentList $DockerExe
    if (Wait-Job $job -Timeout $InfoTimeout) {
        $code = Receive-Job $job
        Remove-Job $job -Force
        return ($code -eq 0)
    }
    Stop-Job $job; Remove-Job $job -Force
    return $false
}

function Show-DockerHelp([string]$why) {
    Write-Host ""
    Write-Host "Docker is not running, or its engine is not answering ($why)." -ForegroundColor Yellow
    Write-Host "  1. Start Docker Desktop from the Start menu."
    Write-Host "  2. Wait until it shows 'Engine running' (bottom-left of its window)."
    Write-Host "  3. Run this command again."
    Write-Host "  If it stays on 'Starting', quit Docker Desktop (tray icon > Quit) and start it again."
}

function Invoke-Docker([string[]]$Arguments) {
    $shown = $Arguments | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }
    Write-Host ("   > docker " + ($shown -join ' ')) -ForegroundColor DarkGray
    if ($DryRun) { return }
    $started = Get-Date
    & $DockerExe @Arguments
    $code = $LASTEXITCODE
    Write-Host ("   ({0:N0}s)" -f ((Get-Date) - $started).TotalSeconds) -ForegroundColor DarkGray
    if ($code -ne 0) { Stop-Pipeline "docker $($Arguments[0]) exited $code" }
}

function Invoke-Pipeline([string[]]$CliArgs, [switch]$Tty) {
    $base = @('run', '--rm')
    if ($Tty) { $base += '-it' }
    $base += @('--network', 'none', '-v', "${Volume}:/app/out", $Image)
    Invoke-Docker ($base + $CliArgs)
}

Set-Location $Root

if ($SelfTest) {
    Write-Host "self test: docker.exe = $DockerExe"
    $v = & $DockerExe --version
    if ($LASTEXITCODE -ne 0) { Write-Host "docker --version failed" -ForegroundColor Red; exit 1 }
    Write-Host "   client : $v"
    Write-Host ("   daemon : " + $(if (Test-DockerDaemon) { "RUNNING" } else { "not running" }))
    Write-Host "self test passed - the docker call path works." -ForegroundColor Green
    exit 0
}

Write-Host "debug907 docker pipeline   repo: $Root   threads: $Threads" -ForegroundColor Green
if ($DryRun) { Write-Host "DRY RUN - nothing will be executed" -ForegroundColor Yellow }
if ($Project) {
    Write-Host "Reminder: pause OneDrive sync - the -Project index is written to the repo's out\ folder." -ForegroundColor Yellow
}

Write-Step "Docker Desktop running"
if (-not $DryRun) {
    if (-not (Test-DockerDaemon)) {
        $exe = Join-Path $env:ProgramFiles 'Docker\Docker\Docker Desktop.exe'
        $running = Get-Process 'Docker Desktop' -ErrorAction SilentlyContinue
        if (-not (Test-Path $exe)) {
            Show-DockerHelp "Docker Desktop is not installed at $exe"
            Stop-Pipeline "Docker is not running"
        }
        if (-not $running) {
            Write-Host "   Docker Desktop is not running: starting it and waiting up to 3 minutes..."
            Start-Process $exe
        } else {
            Write-Host "   Docker Desktop is running but its engine is not answering; waiting up to 3 minutes..."
        }
        $deadline = (Get-Date).AddMinutes(3)
        $up = $false
        while (-not $up -and (Get-Date) -lt $deadline) {
            Start-Sleep -Seconds 5
            Write-Host "." -NoNewline
            $up = Test-DockerDaemon
        }
        Write-Host ""
        if (-not $up) {
            $vm = Get-Process 'vmmem*' -ErrorAction SilentlyContinue
            Show-DockerHelp $(if ($vm) { "engine not answering" } else { "its WSL2 engine did not start" })
            Stop-Pipeline "Docker is not running (see above)"
        }
    }
    Write-Host "   docker is running"
}

if ($SkipBuild) {
    Write-Step "build (skipped, reusing existing image)"
} else {
    Write-Step ("build image" + $(if ($Clean) { " (--no-cache)" } else { "" }) +
                " - bakes model + dataset, runs pytest + selftest")
    $buildArgs = @('build', '-t', $Image, '.')
    if ($Clean) { $buildArgs = @('build', '--no-cache', '-t', $Image, '.') }
    Invoke-Docker $buildArgs
    if (-not $DryRun) {
        $size = & $DockerExe image inspect $Image --format '{{.Size}}'
        Write-Host ("   image size: {0:N2} GB" -f ([double]$size / 1e9))
    }
}

Write-Step "doctor - environment and encoder (network disabled)"
Invoke-Docker @('run', '--rm', '--network', 'none', $Image, 'doctor')

if ($Fresh -and -not $DryRun) {
    & $DockerExe volume rm -f $Volume *> $null
    Write-Host "   (-Fresh) removed volume $Volume - the next step is a full cold encode"
}
Write-Step "real index - full AppsRetrieval corpus, encoded inside the container"
Write-Host "   first run encodes all 8,765 snippets (~45-60 min on $Threads threads); later runs reuse the volume" -ForegroundColor Yellow
Invoke-Pipeline @('index', '--db', 'out/real.db', '--threads', "$Threads")

Write-Step "reproduce - shipped config on the validated 1,000-query sample (must hit 0.6206)"
Invoke-Pipeline @('reproduce', '--db', 'out/real.db', '--threads', "$Threads")

Write-Host ""
if ($DryRun) {
    Write-Host "DRY RUN complete - nothing was executed, nothing was verified." -ForegroundColor Yellow
} else {
    Write-Host "PIPELINE PASSED - the image reproduces the shipped result on real data, offline." -ForegroundColor Green
}

if ($Interactive) {
    Write-Step "interactive search on the real corpus (type :quit to exit)"
    Invoke-Pipeline @('interactive', '--db', 'out/real.db') -Tty
}

if ($Project) {
    $resolved = Resolve-Path $Project -ErrorAction SilentlyContinue
    if (-not $resolved) { Stop-Pipeline "project folder not found: $Project" }
    $ProjectPath = $resolved.Path
    $name = (Split-Path $ProjectPath -Leaf) -replace '[^A-Za-z0-9_-]', '_'
    $db = "out/$name.db"
    New-Item -ItemType Directory -Force -Path $Out | Out-Null

    Write-Step "index your project: $ProjectPath"
    Invoke-Docker @('run', '--rm', '--network', 'none',
                    '-v', "${ProjectPath}:/code:ro", '-v', "${Out}:/app/out",
                    $Image, 'index-repo', '/code', '--out', $db, '--threads', "$Threads")

    Write-Step "interactive search on your project (type :quit to exit)"
    Invoke-Docker @('run', '--rm', '-it', '--network', 'none',
                    '-v', "${Out}:/app/out", $Image, 'interactive', '--db', $db)
}
