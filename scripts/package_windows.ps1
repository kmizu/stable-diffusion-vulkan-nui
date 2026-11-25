[CmdletBinding()]
param(
    [string]$JdkHome = $env:JAVA_HOME,
    [string]$Jar = '',
    [ValidatePattern('^\d{1,3}(\.\d{1,3}){1,2}$')]
    [string]$Version = '0.1',
    [string]$OutputRoot = '',
    [string]$ScalaTorchWorkerDistribution = '',
    [string]$ScalaTorchTokenizerDirectory = '',
    [string]$ScalaTorchJdkHome = '',
    [switch]$PreferScalaTorch,
    [switch]$NativeOnly,
    [switch]$VulkanOnly,
    [string]$VulkanArchive = '',
    [string]$VulkanWorkerJar = '',
    [string]$VulkanVcRedistDir = $env:VCToolsRedistDir
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
if (-not $Jar) { $Jar = Join-Path $repo 'target/stable-diffusion-nui.jar' }
if (-not $OutputRoot) { $OutputRoot = Join-Path $PSScriptRoot "../dist/windows-v$Version" }
if (-not $JdkHome) { throw 'Set JAVA_HOME or pass -JdkHome to a Windows x64 JDK 21 or 25.' }
$jpackage = Join-Path ([IO.Path]::GetFullPath($JdkHome)) 'bin/jpackage.exe'
$java = Join-Path ([IO.Path]::GetFullPath($JdkHome)) 'bin/java.exe'
$jarPath = [IO.Path]::GetFullPath($Jar)
foreach ($file in @($jpackage, $java, $jarPath)) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) { throw "Required file missing: $file" }
}
$start = [Diagnostics.ProcessStartInfo]::new()
$start.FileName = $java
$start.Arguments = '-XshowSettings:properties -version'
$start.UseShellExecute = $false
$start.RedirectStandardError = $true
$start.RedirectStandardOutput = $true
$process = [Diagnostics.Process]::Start($start)
$versionInfo = $process.StandardError.ReadToEnd() + $process.StandardOutput.ReadToEnd()
$process.WaitForExit()
if ($process.ExitCode -ne 0 -or $versionInfo -notmatch 'version "(21|25)[.+"]') { throw 'A JDK 21 or 25 installation is required.' }
if ($versionInfo -notmatch 'os.arch\s*=\s*amd64') { throw 'A Windows x64 JDK is required.' }

if ($VulkanOnly) {
    if ($NativeOnly -or $PreferScalaTorch -or $ScalaTorchWorkerDistribution -or $ScalaTorchJdkHome -or $ScalaTorchTokenizerDirectory) {
        throw 'Choose one distribution profile.'
    }
    if ($versionInfo -notmatch 'version "25[.+"]') { throw 'Vulkan packaging requires a Windows x64 JDK 25.' }
    if (-not $VulkanWorkerJar) { $VulkanWorkerJar = Join-Path $repo 'target/vulkan-worker.jar' }
    if (-not (Test-Path -LiteralPath $VulkanWorkerJar -PathType Leaf)) { throw 'Build vulkanWorker/assembly first.' }
}
$workerDistribution = $null
$nativeJdk = $null
if ($ScalaTorchTokenizerDirectory -and -not $ScalaTorchWorkerDistribution) {
    throw '-ScalaTorchTokenizerDirectory requires -ScalaTorchWorkerDistribution.'
}
if ($ScalaTorchWorkerDistribution -or $ScalaTorchJdkHome -or $PreferScalaTorch -or $NativeOnly) {
    if (-not $ScalaTorchWorkerDistribution -or -not $ScalaTorchJdkHome) {
        throw 'Scala-torch packaging requires both -ScalaTorchWorkerDistribution and -ScalaTorchJdkHome.'
    }
    $workerDistribution = [IO.Path]::GetFullPath($ScalaTorchWorkerDistribution)
    $nativeJdkHome = [IO.Path]::GetFullPath($ScalaTorchJdkHome)
    foreach ($file in @(
        (Join-Path $workerDistribution 'worker-classpath.txt'),
        (Join-Path $nativeJdkHome 'bin/java.exe'),
        (Join-Path $nativeJdkHome 'bin/jlink.exe')
    )) {
        if (-not (Test-Path -LiteralPath $file -PathType Leaf)) { throw "Required native runtime file missing: $file" }
    }
    if (-not (Test-Path -LiteralPath (Join-Path $nativeJdkHome 'jmods') -PathType Container)) { throw 'Java 25 JDK jmods directory is missing.' }
    $manifest = (Get-Content -LiteralPath (Join-Path $workerDistribution 'worker-classpath.txt') -Raw).Trim()
    if (-not $manifest -or $manifest.Contains("`n") -or $manifest.Contains("`r")) { throw 'Native worker manifest must be one nonempty line.' }
    $distRoot = $workerDistribution.TrimEnd('\') + '\'
    foreach ($entry in $manifest.Split([IO.Path]::PathSeparator)) {
        if ([IO.Path]::IsPathRooted($entry)) { throw "Native worker manifest must be relocatable; rooted entry found: $entry" }
        $entryPath = [IO.Path]::GetFullPath((Join-Path $workerDistribution $entry))
        if (-not $entryPath.StartsWith($distRoot, [StringComparison]::OrdinalIgnoreCase)) { throw "Native classpath entry escapes its distribution: $entry" }
        if (-not (Test-Path -LiteralPath $entryPath -PathType Leaf)) { throw "Native worker dependency missing: $entry" }
    }
    $linked = @(Get-ChildItem -LiteralPath $workerDistribution -Force -Recurse | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint })
    if ($linked.Count) { throw "Native worker distribution contains linked files: $($linked[0].FullName)" }
    $nativeJava = Join-Path $nativeJdkHome 'bin/java.exe'
    $nativeStart = [Diagnostics.ProcessStartInfo]::new()
    $nativeStart.FileName = $nativeJava
    $nativeStart.Arguments = '-XshowSettings:properties -version'
    $nativeStart.UseShellExecute = $false
    $nativeStart.RedirectStandardError = $true
    $nativeStart.RedirectStandardOutput = $true
    $nativeProcess = [Diagnostics.Process]::Start($nativeStart)
    $nativeVersion = $nativeProcess.StandardError.ReadToEnd() + $nativeProcess.StandardOutput.ReadToEnd()
    $nativeProcess.WaitForExit()
    if ($nativeProcess.ExitCode -ne 0 -or $nativeVersion -notmatch 'version "25[.+"]') { throw 'Scala-torch worker packaging requires a Java 25 JDK.' }
    if ($nativeVersion -notmatch 'os.arch\s*=\s*amd64') { throw 'A Windows x64 Java 25 JDK is required for scala-torch.' }
}

if ($NativeOnly -or $VulkanOnly) {
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    if ($VulkanOnly) {
        $workerArchive = [IO.Compression.ZipFile]::OpenRead([IO.Path]::GetFullPath($VulkanWorkerJar))
        try {
            foreach ($entry in 'sdfx/vulkan/VulkanWorker.class','sdfx/vulkan/VulkanInference.class') {
                if (-not $workerArchive.GetEntry($entry)) { throw "Vulkan worker jar is missing $entry" }
            }
        } finally { $workerArchive.Dispose() }
    }
    $jarArchive = [IO.Compression.ZipFile]::OpenRead($jarPath)
    try {
        $serviceEntry = $jarArchive.GetEntry('META-INF/services/sdfx.engine.EngineBackendProvider')
        if (-not $serviceEntry) { throw 'Native-only jar has no EngineBackendProvider service descriptor.' }
        $reader = [IO.StreamReader]::new($serviceEntry.Open())
        try { $providers = @($reader.ReadToEnd().Split("`n") | ForEach-Object { $_.Trim() } | Where-Object { $_ -and -not $_.StartsWith('#') }) }
        finally { $reader.Dispose() }
        $expected = if ($VulkanOnly) { 'sdfx.engine.vulkan.VulkanBackendProvider' } else { 'sdfx.engine.scalatorch.ScalaTorchBackendProvider' }
        if ($VulkanOnly -and ($jarArchive.Entries.FullName -match '^(scalatorch|sdfx/(scalatorch|engine/scalatorch))/')) { throw 'Scala-torch present in Vulkan application jar.' }
        if ($providers.Count -ne 1 -or $providers[0] -ne $expected) {
            throw "Native-only jar must contain only $expected; found: $($providers -join ', ')"
        }
        $pythonEntries = @($jarArchive.Entries | Where-Object {
            $_.FullName -match '^sdfx/pythonengine/' -or
            $_.FullName -match '(^|/)Python[^/]*\.(class|tasty)$' -or
            $_.FullName -match '^sdfx/engine/PythonBackendProvider'
        })
        if ($pythonEntries.Count) { throw "Native-only jar contains Python backend code: $($pythonEntries[0].FullName)" }
    }
    finally { $jarArchive.Dispose() }
}

# A unique child prevents overwriting an earlier release or arbitrary user files.
$root = [IO.Path]::GetFullPath($OutputRoot)
[IO.Directory]::CreateDirectory($root) | Out-Null
$run = Join-Path $root ('build-' + [Guid]::NewGuid().ToString('N'))
if (-not ([IO.Path]::GetFullPath($run)).StartsWith($root.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Invalid output path.' }
[IO.Directory]::CreateDirectory($run) | Out-Null
$inputDir = Join-Path $run 'input'
$imageDest = Join-Path $run 'image'
[IO.Directory]::CreateDirectory($inputDir) | Out-Null
[IO.Directory]::CreateDirectory($imageDest) | Out-Null
Copy-Item -LiteralPath $jarPath -Destination (Join-Path $inputDir 'stable-diffusion-nui.jar')

$iconPath = Join-Path $repo 'ui/src/main/resources/sdfx/ui/icons/sdnui.ico'
if (-not (Test-Path -LiteralPath $iconPath -PathType Leaf)) { throw "Application icon missing: $iconPath" }

$jpackageArgs = @('--type', 'app-image', '--name', 'SDNUI', '--description', 'Stable Diffusion NUI', '--app-version', $Version,
    '--icon', $iconPath, '--dest', $imageDest, '--input', $inputDir, '--main-jar', 'stable-diffusion-nui.jar', '--main-class', 'sdfx.ui.Launcher',
    '--java-options', '-Dsdfx.home=$APPDIR\..', '--java-options', '-Dprism.allowhidpi=true', '--java-options', '-Xmx2g', '--java-options', '-Dfile.encoding=UTF-8',
    '--java-options', '--enable-native-access=ALL-UNNAMED')
if ($PreferScalaTorch -or $NativeOnly) { $jpackageArgs += @('--java-options', '-Dsdfx.engine.backend=scala-torch') }
& $jpackage @jpackageArgs
if ($LASTEXITCODE -ne 0) { throw "jpackage failed (exit $LASTEXITCODE); staging retained at $run" }
$image = Join-Path $imageDest 'SDNUI'
foreach ($required in @('SDNUI.exe', 'app/stable-diffusion-nui.jar', 'runtime/legal')) {
    if (-not (Test-Path -LiteralPath (Join-Path $image $required))) { throw "Missing packaged component: $required" }
}
if ($VulkanOnly) {
    $nativeRoot = Join-Path $image 'vulkan'
    $nativeBin = Join-Path $nativeRoot 'bin'
    [IO.Directory]::CreateDirectory($nativeBin) | Out-Null
    $asset = 'sd-master-f296cb0-bin-win-vulkan-x64.zip'
    if (-not $VulkanArchive) {
        $VulkanArchive = Join-Path $run $asset
        Invoke-WebRequest -UseBasicParsing -Uri "https://github.com/leejet/stable-diffusion.cpp/releases/download/master-1129-f296cb0/$asset" -OutFile $VulkanArchive
    }
    if ((Get-FileHash -LiteralPath $VulkanArchive).Hash -ne '00f1093279b64a1a4b3a14d8cb7b6fe1920cb8eda31f2fde30e17546fbf8e78d') { throw 'Vulkan archive checksum mismatch.' }
    $upstream = Join-Path $run 'upstream'
    Expand-Archive -LiteralPath $VulkanArchive -DestinationPath $upstream
    Get-ChildItem -LiteralPath $upstream -File | Where-Object { $_.Extension -in '.dll','.txt' } | Copy-Item -Destination $nativeBin
    if (-not $VulkanVcRedistDir) {
        $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
        if (Test-Path -LiteralPath $vswhere) {
            $vs = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
            if ($vs) {
                $redist = Get-ChildItem -Directory (Join-Path $vs 'VC/Redist/MSVC') | Where-Object { $_.Name -match '^14\.[\d.]+$' } |
                    Sort-Object { [version]$_.Name } -Descending | Select-Object -First 1
                if ($redist) { $VulkanVcRedistDir = $redist.FullName }
            }
        }
    }
    if (-not $VulkanVcRedistDir) { throw 'Pass the Visual C++ redistributable directory (VC/Redist/MSVC/<version>) with -VulkanVcRedistDir.' }
    foreach ($name in 'msvcp140.dll','msvcp140_codecvt_ids.dll','vcruntime140.dll','vcruntime140_1.dll','vcomp140.dll') {
        $component = if ($name -eq 'vcomp140.dll') { 'Microsoft.VC143.OpenMP' } else { 'Microsoft.VC143.CRT' }
        Copy-Item -LiteralPath (Join-Path $VulkanVcRedistDir "x64/$component/$name") -Destination $nativeBin
    }
    Copy-Item -LiteralPath $VulkanWorkerJar -Destination (Join-Path $nativeRoot 'vulkan-worker.jar')
    & (Join-Path $JdkHome 'bin/jlink.exe') --add-modules java.base,java.desktop,jdk.unsupported --strip-debug --no-header-files --no-man-pages --output (Join-Path $nativeRoot 'jre')
    if ($LASTEXITCODE -ne 0) { throw 'Failed to link the Vulkan worker runtime.' }
}
if ($workerDistribution) {
    $nativeRoot = Join-Path $image 'scala-torch'
    [IO.Directory]::CreateDirectory($nativeRoot) | Out-Null
    Get-ChildItem -LiteralPath $workerDistribution -Force | Copy-Item -Destination $nativeRoot -Recurse -Force
    if ($ScalaTorchTokenizerDirectory) {
        $tokenizerSource = [IO.Path]::GetFullPath($ScalaTorchTokenizerDirectory)
        $tokenizerDestination = Join-Path $nativeRoot 'tokenizer'
        [IO.Directory]::CreateDirectory($tokenizerDestination) | Out-Null
        foreach ($name in @('vocab.json', 'merges.txt')) {
            $source = Join-Path $tokenizerSource $name
            if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { throw "Tokenizer file missing: $source" }
            $item = Get-Item -LiteralPath $source
            if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Refusing linked tokenizer file: $source" }
            Copy-Item -LiteralPath $source -Destination (Join-Path $tokenizerDestination $name)
        }
    }
    $jlink = Join-Path $nativeJdkHome 'bin/jlink.exe'
    $nativeRuntime = Join-Path $nativeRoot 'jre'
    & $jlink --module-path (Join-Path $nativeJdkHome 'jmods') --add-modules ALL-MODULE-PATH --strip-debug --no-header-files --no-man-pages --compress=2 --output $nativeRuntime
    if ($LASTEXITCODE -ne 0) { throw "jlink failed (exit $LASTEXITCODE); staging retained at $run" }
    $nativeJavaExe = Join-Path $nativeRuntime 'bin/java.exe'
    if (-not (Test-Path -LiteralPath $nativeJavaExe -PathType Leaf)) { throw 'Bundled Java 25 runtime is missing.' }
}

# Only tracked, explicitly allowed source and licence files are copied: no models, environments,
# credentials, caches, settings, or output files from the developer's checkout. The single model
# exception is the pinned NudeNet NSFW Censor detector models/nudenet/best.onnx (AGPL-3.0, from
# w-e-w/sd-webui-nudenet-nsfw-censor@1fb21cf); its SHA-256 is verified before it is copied.
# The CLIP Interrogator label lists (python/sdfx/data/clip_interrogator/*.txt, MIT upstream lists) are plain data the engine reads at run time.
$tracked = @(git -C $repo ls-files -- python/sdfx models licenses)
if ($LASTEXITCODE -ne 0) { throw 'git ls-files failed; a Git checkout is required.' }
$payload = @($tracked | Where-Object { $_ -match '^python/sdfx/.+\.py$|^python/sdfx/waifu2x_assets\.json$|^python/sdfx/data/model_keyword/(model|lora)-keyword\.txt$|^python/sdfx/data/clip_interrogator/[a-z]+\.txt$|^python/sdfx/data/clip_interrogator/openclip_registry_licenses\.json$|^python/sdfx/.+/LICENSE$|^models/.+/README\.txt$|^models/nudenet/best\.onnx$|^licenses/[A-Za-z0-9_.-]+\.txt$' })
if (-not ($NativeOnly -or $VulkanOnly) -and -not ($payload -contains 'python/sdfx/__main__.py')) { throw 'Python source files missing from Git index.' }
if ($NativeOnly -or $VulkanOnly) { $payload = @($payload | Where-Object { -not $_.StartsWith('python/') }) }
$nudenet = 'models/nudenet/best.onnx'
if (-not ($payload -contains $nudenet)) { throw "Pinned NudeNet detector missing from Git index: $nudenet" }
$nudenetHash = (Get-FileHash -LiteralPath (Join-Path $repo $nudenet) -Algorithm SHA256).Hash.ToLowerInvariant()
if ($nudenetHash -ne '9832f15515bdb06bcb5a77beb60bc8ea54439bd7ecbaac46dac3b760b3dd13cc') { throw "NudeNet detector hash mismatch: $nudenetHash" }
foreach ($relative in $payload) {
    $source = Join-Path $repo $relative
    $item = Get-Item -LiteralPath $source
    if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Refusing linked source: $relative" }
    $destination = Join-Path $image $relative
    [IO.Directory]::CreateDirectory((Split-Path -Parent $destination)) | Out-Null
    Copy-Item -LiteralPath $source -Destination $destination
}
$appFiles = @('LICENSE', 'THIRD_PARTY_NOTICES.md', 'DISTRIBUTION_NOTICES.md', '.mcp.json.example', 'mcp/README.md', 'docs/mcp-generation-metadata.md', 'docs/mcp-agent-verification.md', 'docs/batch-face-checkpoint-ja.md', 'docs/batch-face-audit-ja.md')
if (-not ($NativeOnly -or $VulkanOnly)) { $appFiles += @('python/requirements.txt', 'python/requirements-torch.txt', 'python/requirements-dev.txt') }
foreach ($relative in $appFiles) {
    $source = Join-Path $repo $relative
    $destination = Join-Path $image $relative
    [IO.Directory]::CreateDirectory((Split-Path -Parent $destination)) | Out-Null
    Copy-Item -LiteralPath $source -Destination $destination
}
if (-not $VulkanOnly) {
    Copy-Item -LiteralPath (Join-Path $repo 'docs/WINDOWS_RELEASE.md') -Destination (Join-Path $image 'WINDOWS_RELEASE.md')
}
$zipName = "SDNUI-v$Version-windows-x64.zip"
if ($VulkanOnly) {
    $zipName = "SDNUI-v$Version-windows-vulkan-x64.zip"
    Copy-Item -LiteralPath (Join-Path $repo 'docs/WINDOWS_VULKAN.md') -Destination (Join-Path $image 'WINDOWS_VULKAN.md')
}
$zip = Join-Path $run $zipName
Compress-Archive -LiteralPath $image -DestinationPath $zip -CompressionLevel Optimal
$hash = (Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash.ToLowerInvariant()
[IO.File]::WriteAllText((Join-Path $run 'SHA256SUMS.txt'), "$hash  $zipName`n", [Text.UTF8Encoding]::new($false))
Write-Host "Package: $zip"
Write-Host "SHA-256: $hash"
Write-Host "Staging retained: $run"
