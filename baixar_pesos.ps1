$ErrorActionPreference = 'Stop'
$projectPath = $PSScriptRoot
$weightsDirectory = Join-Path $projectPath 'models\pretrained\checkpoints'
New-Item -ItemType Directory -Path $weightsDirectory -Force | Out-Null
$weightsPath = Join-Path $weightsDirectory 'efficientnet_b0_rwightman-7f5810bc.pth'
$partialPath = $weightsPath + '.partial'
$weightsUrl = 'https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth'
Write-Host 'Baixando pesos oficiais ImageNet para EfficientNetB0...'
Invoke-WebRequest -Uri $weightsUrl -OutFile $partialPath
$digest = (Get-FileHash -LiteralPath $partialPath -Algorithm SHA256).Hash.ToLowerInvariant()
if (-not $digest.StartsWith('7f5810bc')) {
    throw 'Hash dos pesos divergente. O arquivo parcial foi preservado para inspeção.'
}
Move-Item -LiteralPath $partialPath -Destination $weightsPath -Force
Write-Host 'Pesos baixados e hash conferido.'
Write-Host $weightsPath
