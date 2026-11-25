[CmdletBinding()]
param([Parameter(Mandatory)][string]$Destination,[Parameter(Mandatory)][string]$JdkHome)
$ErrorActionPreference='Stop'
$root=Join-Path $PSScriptRoot '../native/custom-code'
$lib=Join-Path $Destination 'lib'
[IO.Directory]::CreateDirectory($lib) | Out-Null
foreach($line in Get-Content -LiteralPath "$root/dependencies.sha256") {
    $hash,$relative=$line -split ' ',2
    $name=[IO.Path]::GetFileName($relative)
    $target=Join-Path $lib $name
    if (!(Test-Path -LiteralPath $target)) {
        $cached=Join-Path $env:LOCALAPPDATA "Coursier/cache/v1/https/repo.maven.apache.org/maven2/$relative"
        if (Test-Path -LiteralPath $cached) { Copy-Item -LiteralPath $cached -Destination $target }
        else { Invoke-WebRequest -Uri "https://repo.maven.apache.org/maven2/$relative" -OutFile $target }
    }
    if ((Get-FileHash -LiteralPath $target).Hash -ne $hash) {throw "Digest mismatch for $name"}
}
& "$JdkHome/bin/javac.exe" -cp "$lib/*" -d $Destination "$root/VulkanScriptHost.java"
if ($LASTEXITCODE -ne 0) {throw 'Custom script host compilation failed'}
Copy-Item -LiteralPath "$root/vulkan_script.py","$root/dependencies.sha256" -Destination $Destination -Force
