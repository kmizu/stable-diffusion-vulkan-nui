[CmdletBinding()]
param([Parameter(Mandatory)][string]$Destination,
      [string]$Cache = (Join-Path $PSScriptRoot '../target/sdv-native-build'))
$ErrorActionPreference = 'Stop'
$revision = 'f296cb047ea358620167712e0b58ed79aae0791b'
$Cache = [IO.Path]::GetFullPath($Cache)
$source = Join-Path $Cache 'source'
$build = Join-Path $Cache 'build'
$patch = (Resolve-Path (Join-Path $PSScriptRoot '../native/stable-diffusion/clip-vision.patch')).Path
function Run([string]$exe, [string[]]$arguments) {
    & $exe @arguments
    if ($LASTEXITCODE -ne 0) { throw "$exe failed ($LASTEXITCODE)" }
}
[IO.Directory]::CreateDirectory($Cache) | Out-Null
if (!(Test-Path "$source/.git")) {
    Run git @('init',$source)
    Run git @('-C',$source,'remote','add','origin','https://github.com/leejet/stable-diffusion.cpp.git')
}
if ((git -C $source rev-parse --verify --quiet HEAD) -ne $revision) {
    Run git @('-C',$source,'fetch','--depth=1','origin',$revision)
    Run git @('-C',$source,'checkout','--detach',$revision)
}
Run git @('-C',$source,'submodule','update','--init','--depth','1','ggml')
$patchDir=Split-Path $patch
$patches = @('clip-vision.patch','svd.patch','conditioning-stats.patch','mask-conditioning.patch','unet-hooks.patch','dynamic-threshold.patch','clip-io-options.patch','clip-io-tokens.patch','multidiffusion.patch','latent-modifier.patch','control-resized-features.patch','negpip.patch','fooocus-inpaint.patch','cd-tuner.patch','tgate.patch','deepcache.patch','fast-cache.patch','reference.patch')
# Compose the patches against the pinned Git tree. Only rewrite sources whose blob changes.
$oldIndex = $env:GIT_INDEX_FILE
$patchIndex = Join-Path $Cache ("patches-" + [guid]::NewGuid() + ".index")
try {
    $env:GIT_INDEX_FILE = $patchIndex
    Run git @('-C',$source,'read-tree',$revision)
    foreach ($name in $patches) {
        Run git @('-C',$source,'apply','--cached',(Join-Path $patchDir $name))
    }
    $files = @(git -C $source diff --cached --name-only $revision)
    foreach ($file in $files) {
        $expected = git -C $source rev-parse ":$file"
        $actual = if (Test-Path -LiteralPath (Join-Path $source $file)) { git -C $source hash-object "--path=$file" $file } else { '' }
        if ($actual -ne $expected) { Run git @('-C',$source,'checkout-index','--force','--',$file) }
    }
} finally {
    $env:GIT_INDEX_FILE = $oldIndex
    if (Test-Path -LiteralPath $patchIndex) { Remove-Item -LiteralPath $patchIndex }
}
$cmakeCommand = Get-Command cmake.exe -ErrorAction SilentlyContinue
$cmake = if ($cmakeCommand) { $cmakeCommand.Source } else { $null }
if (!$cmake) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
    $vs = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    $cmake = Join-Path $vs 'Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe'
}
Run $cmake @('-S',$source,'-B',$build,'-G','Visual Studio 17 2022','-A','x64','-DSD_BUILD_SHARED_LIBS=ON','-DSD_BUILD_SHARED_GGML_LIB=ON','-DSD_BUILD_EXAMPLES=OFF','-DSD_WEBP=OFF','-DSD_WEBM=OFF','-DGGML_BACKEND_DL=ON','-DGGML_NATIVE=OFF')
Run $cmake @('--build',$build,'--config','Release','--target','stable-diffusion','--parallel','4')
[IO.Directory]::CreateDirectory($Destination) | Out-Null
Copy-Item -LiteralPath "$build/bin/Release/stable-diffusion.dll" -Destination $Destination -Force
