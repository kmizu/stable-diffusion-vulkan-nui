[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$Destination,
    [switch]$IncludeWeights,
    [switch]$IncludeCharacterWeights,
    [switch]$IncludeBlip,
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
# The existing bundled NudeNet detector is also needed for Vulkan preview censoring.
$converter = Get-Sdk 'pnnx/pnnx' 'pnnx-20261020-windows' 'e4f91348a399a0060a6e7dd323d40427e1cbdcab67469cf1904b5d09b23606c5'
$nudeSource = Join-Path $repo 'models/nudenet/best.onnx'
Confirm-Hash $nudeSource '9832f15515bdb06bcb5a77beb60bc8ea54439bd7ecbaac46dac3b760b3dd13cc'
$nudeWork = Join-Path $Cache 'nudenet'
[IO.Directory]::CreateDirectory($nudeWork) | Out-Null
if (-not [IO.File]::Exists("$nudeWork/nudenet.ncnn.bin")) {
    Copy-Item -LiteralPath $nudeSource -Destination "$nudeWork/nudenet.onnx" -Force
    Push-Location -LiteralPath $nudeWork
    try {
        & "$converter/pnnx.exe" nudenet.onnx 'inputshape=[1,3,320,320]' 'fp16=0'
        if ($LASTEXITCODE -ne 0) { throw 'NudeNet conversion failed.' }
    } finally { Pop-Location }
}
Confirm-Hash "$nudeWork/nudenet.ncnn.param" '052b03a70278114f47c953468206f852f9b3c7f9756f5b86a786be80f8a5d711'
Confirm-Hash "$nudeWork/nudenet.ncnn.bin" '8a34c4116a77a619d36ec5a5d8863754dc44ec59a6510bce5fecac6a61ad5485'
$nudeDest = Join-Path $Destination 'nsfw'
[IO.Directory]::CreateDirectory($nudeDest) | Out-Null
Copy-Item -LiteralPath "$nudeWork/nudenet.ncnn.param","$nudeWork/nudenet.ncnn.bin" -Destination $nudeDest -Force
Copy-Item -LiteralPath "$repo/licenses/notAI-tech__NudeNet.txt","$repo/licenses/w-e-w__sd-webui-nudenet-nsfw-censor.txt" -Destination $nudeDest -Force
if ($IncludeWeights -or $IncludeCharacterWeights) {
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
if ($IncludeCharacterWeights) {
    $work = Join-Path $Cache 'chara-searcher'
    $bundle = Join-Path $Destination 'chara-searcher'
    [IO.Directory]::CreateDirectory($work) | Out-Null
    [IO.Directory]::CreateDirectory($bundle) | Out-Null
    $assets = @(
        @{Name='person'; Repo='deepghs/anime_person_detection'; Revision='e39c744c22432ad01f91dd254fe2b02c8d878b8c'; File='person_detect_v1.1_m/model.onnx'; Size=640;
          Original='c80365a98ca8388fd2f01fd5abea0439e92265bceceb048bef38b0849303467c'; Param='a7324706595da41a484aac7c140a68c60ad9805457e1c084454fa6eaa041bb17'; Bin='e0d4e89c822c1915b431abab6b59945911a99b898674bc373568276e8d638bf3'}
        @{Name='seg'; Repo='skytnt/anime-seg'; Revision='493cb60893f47441b26ec4fb9a306bce9e342982'; File='isnetis.onnx'; Size=1024;
          Original='f15622d853e8260172812b657053460e20806f04b9e05147d49af7bed31a6e99'; Param='2e6dbb845310177b3ce3f6858d8f876aba8e20a10c4118a324ff60f65bc269d6'; Bin='3823cf7a137b09410b7e9d356617791f15a9a9a32bbf8ef4fd021c3985390abb'}
    )
    foreach ($asset in $assets) {
        $name = $asset.Name
        $cacheRoot = $env:SDFX_CHARA_CACHE
        $source = if ($cacheRoot) { Join-Path $cacheRoot ($asset.Repo.Replace('/','__')+'/'+$asset.Revision+'/'+[IO.Path]::GetFileName($asset.File)) }
        if ($source -and (Test-Path -LiteralPath $source)) {
            Confirm-Hash $source $asset.Original
        } else {
            $source = Get-Asset "https://huggingface.co/$($asset.Repo)/resolve/$($asset.Revision)/$($asset.File)" "$name.onnx" $asset.Original
        }
        Copy-Item -LiteralPath $source -Destination "$work/$name.onnx" -Force
        Push-Location -LiteralPath $work
        try {
            & "$converter/pnnx.exe" "$name.onnx" "inputshape=[1,3,$($asset.Size),$($asset.Size)]" 'fp16=0'
            if ($LASTEXITCODE -ne 0) { throw "Conversion failed for $name" }
        } finally { Pop-Location }
        Confirm-Hash "$work/$name.ncnn.param" $asset.Param
        Confirm-Hash "$work/$name.ncnn.bin" $asset.Bin
        Copy-Item -LiteralPath "$work/$name.ncnn.param","$work/$name.ncnn.bin" -Destination $bundle
        Copy-Item -LiteralPath (Join-Path "$repo/licenses" ($asset.Repo.Replace('/','__')+'.txt')) -Destination $bundle
    }
}
if ($IncludeBlip) {
    $source = $env:SDFX_VULKAN_BLIP_DIR
    if (-not $source) { throw 'SDFX_VULKAN_BLIP_DIR must point to the output of scripts/convert_vulkan_blip.py.' }
    $bundle = Join-Path $Destination 'blip-base'
    [IO.Directory]::CreateDirectory($bundle) | Out-Null
    $files = @{
        'vision.ncnn.param' = '7da371282df5d9ea6d29b661eceadde5e4c83eff91356937d1f020d3de270be6'
        'vision.ncnn.bin' = '75fcfd12f40e59a8bd4a7e43f9ce38935da7d49ec92f84e63d9c4eaa1aecf7fd'
        'blip_decoder.ncnn.param' = 'd253631a0113221d1cb48c95ffa9115c43903ba8e00732f88f4228dccb5808f3'
        'blip_decoder.ncnn.bin' = '6dce4d756ee2f9cf097833cb0a246cbca726bd659650d8a34ce54418091f6519'
        'vocab.txt' = '07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3'
    }
    foreach ($file in $files.Keys) {
        $path = Join-Path $source $file
        Confirm-Hash $path $files[$file]
        Copy-Item -LiteralPath $path -Destination $bundle
    }
    Copy-Item -LiteralPath "$repo/licenses/Salesforce__blip-image-captioning-base.txt" -Destination $bundle
}
