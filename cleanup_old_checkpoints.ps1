# Cleanup old checkpoints, keep only 5 most recent per model

$checkpointsDir = "C:\Users\clavi\Desktop\для ии\checkpoints"

Write-Host "Scanning checkpoints..." -ForegroundColor Yellow

# Get all model directories
Get-ChildItem $checkpointsDir -Directory | ForEach-Object {
    $modelDir = $_
    $modelName = $_.Name
    
    Write-Host "`nModel: $modelName" -ForegroundColor Cyan
    
    # Get all checkpoint files for this model, sorted by LastWriteTime (newest first)
    $checkpoints = Get-ChildItem $modelDir.FullName -File -Filter "model_*.pt" | Sort-Object LastWriteTime -Descending
    
    $total = $checkpoints.Count
    Write-Host "  Found $total checkpoints"
    
    if ($total -gt 5) {
        # Keep only 5 newest, delete the rest
        $toDelete = $checkpoints | Select-Object -Skip 5
        $deleteCount = $toDelete.Count
        
        Write-Host "  Keeping 5 newest, deleting $deleteCount old checkpoints..." -ForegroundColor Yellow
        
        $toDelete | ForEach-Object {
            Write-Host "    Deleting: $($_.Name)" -ForegroundColor Red
            Remove-Item $_.FullName -Force
        }
        
        Write-Host "  Done! Freed up space." -ForegroundColor Green
    } else {
        Write-Host "  Only $total checkpoints, keeping all." -ForegroundColor Green
    }
}

Write-Host "`nCleanup complete!" -ForegroundColor Green
