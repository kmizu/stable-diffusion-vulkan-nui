[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$Destination,
    [switch]$IncludeWeights,
    [string]$Cache = (Join-Path $PSScriptRoot '../target/ncnn-sdk'),
    [string]$SourceModelDir = $env:SDFX_WD_V3_MODEL_DIR
)
$ErrorActionPreference = 'Stop'
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$Cache = [IO.Path]::GetFullPath($Cache)
$Destination = [IO.Path]::GetFullPath($Destination)
[IO.Directory]::CreateDirectory($Cache) | Out-Null
[IO.Directory]::CreateDirectory($Destination) | Out-Null
function Confirm-Hash([string]$File, [string]$Hash) {
    if ((Get-FileHash -LiteralPath $File -Algorithm SHA256).Hash -ne $Hash) { throw "Incorrect SHA-256 for $File" }
}
function Get-Asset([string]$Url, [string]$Name, [string]$Digest) {
    $file = Join-Path $Cache $Name
    if (-not [IO.File]::Exists($file)) {
        $temp = "$file.$([guid]::NewGuid()).part"
        try {
            Invoke-WebRequest -Uri $Url -OutFile $temp
            Confirm-Hash $temp $Digest
            [IO.File]::Move($temp,$file)
        } finally { if ([IO.File]::Exists($temp)) { [IO.File]::Delete($temp) } }
    }
    Confirm-Hash $file $Digest
    return $file
}
function Get-Sdk([string]$Project,[string]$Name,[string]$Digest) {
    $zip = Get-Asset "https://github.com/$Project/releases/download/20261020/$Name.zip" "$Name.zip" $Digest
    $dir = Join-Path $Cache $Name
    if (-not [IO.Directory]::Exists($dir)) { Expand-Archive -LiteralPath $zip -DestinationPath $Cache }
    return $dir
}
$sdk = Get-Sdk 'Tencent/ncnn' 'ncnn-20261020-windows-vs2022-shared' '7f39107af5ee4c1e657b8fd01e8b3cad6c0ab7fdbb0713532f01937134285b3d'
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
$vs = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw 'Visual Studio x64 C++ Build Tools are required to build the ncnn bridge.' }
Import-Module (Join-Path $vs 'Common7/Tools/Microsoft.VisualStudio.DevShell.dll')
Enter-VsDevShell -VsInstallPath $vs -SkipAutomaticLocation -DevCmdArguments '-arch=amd64 -host_arch=amd64' | Out-Null
$obj = Join-Path $Cache 'sdv-ncnn.obj'
$exe = Join-Path $Destination 'sdv-ncnn.exe'
& cl.exe /nologo /O2 /EHsc /std:c++17 /MD /Brepro "/Fo$obj" "/I$sdk/x64/include/ncnn" (Join-Path $repo 'native/ncnn/sdv-ncnn.cpp') /link "/OUT:$exe" "$sdk/x64/lib/ncnn.lib" /Brepro
if ($LASTEXITCODE -ne 0) { throw 'ncnn bridge compilation failed.' }
Copy-Item -LiteralPath "$sdk/x64/bin/ncnn.dll" -Destination $Destination
Copy-Item -LiteralPath (Join-Path $repo 'licenses/tencent__ncnn.txt') -Destination $Destination
if ($IncludeWeights) {
    $version = '7f6b584d0bd3f55c4531f14ba3d4761b2bccdc0f'
    $hashModel = '35f23693620b668f4d53fd3c62bf65e40af739bc52c7eb0fbc49258b58d065b6'
    $hashTags = '298633d94d0031d2081c0893f29c82eab7f0df00b08483ba8f29d1e979441217'
    $modelBase = "https://huggingface.co/SmilingWolf/wd-vit-tagger-v3/resolve/$version"
    if ($SourceModelDir) {
        $model = Join-Path $SourceModelDir 'model.onnx'
        $tags = Join-Path $SourceModelDir 'selected_tags.csv'
        Confirm-Hash $model $hashModel
        Confirm-Hash $tags $hashTags
    } else {
        $model = Get-Asset "$modelBase/model.onnx" 'wd-vit-v3.onnx' $hashModel
        $tags = Get-Asset "$modelBase/selected_tags.csv" 'selected_tags.csv' $hashTags
    }
    $converter = Get-Sdk 'pnnx/pnnx' 'pnnx-20261020-windows' 'e4f91348a399a0060a6e7dd323d40427e1cbdcab67469cf1904b5d09b23606c5'
    $work = Join-Path $Cache 'wd-vit-v3'
    [IO.Directory]::CreateDirectory($work) | Out-Null
    if (-not [IO.File]::Exists("$work/wd.ncnn.bin")) {
        Copy-Item -LiteralPath $model -Destination "$work/wd.onnx" -Force
        Push-Location -LiteralPath $work
        try {
            & "$converter/pnnx.exe" wd.onnx 'inputshape=[1,448,448,3]' 'fp16=0'
            if ($LASTEXITCODE -ne 0) { throw 'WD ONNX conversion failed.' }
        } finally { Pop-Location }
    }
    Confirm-Hash "$work/wd.ncnn.param" '17b3c57ffc6877ae8865004ecfb3b82c59d85adb51f460a3b1932f568cc5d293'
    Confirm-Hash "$work/wd.ncnn.bin" '8c93f8b15a7ec2a32ede1edcdf695d8fcd339dca9c73bfed74c14918a5201d27'
    $bundle = Join-Path $Destination 'wd-vit-v3'
    [IO.Directory]::CreateDirectory($bundle) | Out-Null
    Copy-Item -LiteralPath "$work/wd.ncnn.param","$work/wd.ncnn.bin",$tags -Destination $bundle
    Copy-Item -LiteralPath "$repo/licenses/SmilingWolf__wd-vit-tagger-v3.txt" -Destination $bundle
}
