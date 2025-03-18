// Main functionality for Paint by Numbers app
document.addEventListener('DOMContentLoaded', function() {
    // Initialize UI
    toggleOptions();
    
    // Setup form submission
    const form = document.getElementById('convert-form');
    if (form) {
        form.addEventListener('submit', function() {
            document.getElementById('loading').style.display = 'block';
        });
    }
});

function toggleOptions() {
    const mode = document.getElementById('processing_mode').value;
    document.getElementById('preset-options').style.display = (mode === 'preset') ? 'block' : 'none';
    document.getElementById('custom-options').style.display = (mode === 'custom') ? 'block' : 'none';
}