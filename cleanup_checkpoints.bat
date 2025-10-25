@echo off
chcp 65001 >nul 2>&1
echo.
echo ================================================================
echo          Cleaning Old Checkpoints
echo ================================================================
echo.
echo Current disk space:
powershell -Command "Get-PSDrive C | Select-Object @{Name='Free(GB)';Expression={[math]::Round($_.Free/1GB,2)}}"
echo.
echo Checkpoint folder size:
cd /d "%~dp0"
powershell -Command "Get-ChildItem '.\checkpoints' -Recurse -File | Measure-Object -Property Length -Sum | Select-Object @{Name='Size(GB)';Expression={[math]::Round($_.Sum/1GB,2)}}, Count"
echo.
echo This will keep only 5 most recent checkpoints per model.
echo All older checkpoints will be DELETED!
echo.
pause
echo.
echo Cleaning...
powershell -ExecutionPolicy Bypass -Command "& {cd '%~dp0'; Get-ChildItem '.\checkpoints' -Directory | ForEach-Object {$model=$_.Name; Write-Host \"Model: $model\"; $files=Get-ChildItem $_.FullName -File -Filter 'model_*.pt' | Sort-Object LastWriteTime -Descending; $count=$files.Count; Write-Host \"  Found $count checkpoints\"; if($count -gt 5){$toDelete=$files | Select-Object -Skip 5; Write-Host \"  Deleting $($toDelete.Count) old files...\"; $toDelete | Remove-Item -Force; Write-Host \"  Done!\" -ForegroundColor Green}else{Write-Host \"  Keeping all\" -ForegroundColor Green}}}"
echo.
echo Done!
echo.
echo New disk space:
powershell -Command "Get-PSDrive C | Select-Object @{Name='Free(GB)';Expression={[math]::Round($_.Free/1GB,2)}}"
echo.
pause
