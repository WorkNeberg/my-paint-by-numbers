// Main functionality for Paint by Numbers app
document.addEventListener('DOMContentLoaded', function() {
    // Initialize UI
    toggleOptions();
    
    // Setup form submission
    const form = document.getElementById('convert-form');
    if (form) {
        form.addEventListener('submit', function() {
            document.getElementById('loading').classList.remove('hidden');
        });
    }
    
    // Setup image preview
    const imageInput = document.getElementById('image');
    const imagePreview = document.getElementById('image-preview');
    const previewContainer = document.getElementById('preview-container');
    
    if (imageInput && imagePreview && previewContainer) {
        imageInput.addEventListener('change', function() {
            if (this.files && this.files[0]) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    imagePreview.src = e.target.result;
                    previewContainer.classList.remove('hidden');
                };
                reader.readAsDataURL(this.files[0]);
            }
        });
    }
});

function toggleOptions() {
    const mode = document.getElementById('processing_mode').value;
    const presetOptions = document.getElementById('preset-options');
    const customOptions = document.getElementById('custom-options');
    
    if (presetOptions && customOptions) {
        presetOptions.classList.toggle('hidden', mode !== 'preset');
        customOptions.classList.toggle('hidden', mode !== 'custom');
    }
}