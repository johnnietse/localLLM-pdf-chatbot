<#
.SYNOPSIS
Downloads GGUF model files from Hugging Face
#>

$ErrorActionPreference = "Stop"

# Configuration
$ModelDir = "models"
$Models = @{
    "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    "mistral-7b-instruct-v0.1.Q4_K_M.gguf" = "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
}

# Create models directory if it doesn't exist
if (-not (Test-Path -Path $ModelDir)) {
    New-Item -ItemType Directory -Path $ModelDir | Out-Null
}

# Download function with progress
function Download-File {
    param (
        [string]$Url,
        [string]$FileName
    )
    
    $DestPath = Join-Path -Path $ModelDir -ChildPath $FileName
    
    if (Test-Path -Path $DestPath) {
        Write-Host "[OK] $FileName already exists" -ForegroundColor Green
        return
    }

    Write-Host "[Downloading] $FileName..." -ForegroundColor Cyan
    try {
        # Download with progress display
        $ProgressPreference = 'SilentlyContinue' # Speeds up download
        Invoke-WebRequest -Uri $Url -OutFile $DestPath -UseBasicParsing
        
        Write-Host "[OK] Downloaded $FileName" -ForegroundColor Green
    }
    catch {
        Write-Host "[ERROR] Failed to download $FileName" -ForegroundColor Red
        Write-Host "Error details: $_" -ForegroundColor Red
        if (Test-Path -Path $DestPath) { Remove-Item -Path $DestPath }
        throw
    }
}

# Main download process
try {
    foreach ($Model in $Models.GetEnumerator()) {
        Download-File -Url $Model.Value -FileName $Model.Key
    }
    Write-Host "[SUCCESS] All models downloaded to $ModelDir\" -ForegroundColor Green
}
catch {
    Write-Host "[FAILED] Script failed: $_" -ForegroundColor Red
    exit 1
}