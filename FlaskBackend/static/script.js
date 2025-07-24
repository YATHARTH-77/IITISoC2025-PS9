// DOM Elements
const uploadSection = document.getElementById('upload-section');
const previewSection = document.getElementById('preview-section');
const fileInput = document.getElementById('file-input');
const previewImage = document.getElementById('preview-image');
const fileName = document.getElementById('file-name');
const removeImageBtn = document.getElementById('remove-image');
const processBtn = document.getElementById('process-btn');
const processText = document.getElementById('process-text');
const processWarning = document.getElementById('process-warning');
const chooseFileBtn = document.querySelector('.choose-file-btn');
const loader = document.getElementById('loader');

// State
let selectedFile = null;
let selectedLanguage = '';
let isProcessing = false;

// File handling function
function handleFile(file) {
    if (file && (file.type === 'image/jpeg' || file.type === 'image/png')) {
        selectedFile = file;
        
        // Create file URL and display preview
        const fileURL = URL.createObjectURL(file);
        previewImage.src = fileURL;
        fileName.textContent = file.name;
        
        // Show preview, hide upload
        uploadSection.style.display = 'none';
        previewSection.style.display = 'block';
        
        // Update process button state
        updateProcessButton();
        
        console.log('File accepted:', file.name);
    } else {
        alert('Please upload a valid image file (JPEG or PNG).');
    }
}

// Remove image function
function removeImage() {
    if (previewImage.src) {
        URL.revokeObjectURL(previewImage.src);
    }
    
    selectedFile = null;
    previewImage.src = '';
    fileName.textContent = '';
    
    // Show upload, hide preview
    uploadSection.style.display = 'block';
    previewSection.style.display = 'none';
    
    // Reset file input
    fileInput.value = '';
    
    // Update process button state
    updateProcessButton();
}

// Update process button state
function updateProcessButton() {
    const canProcess = selectedFile && selectedLanguage && !isProcessing;
    
    processBtn.disabled = !canProcess;
    
    if (canProcess) {
        processWarning.classList.add('hidden');
        loader.style.display = ''; // Show loader when processing
    } else if (!isProcessing) {
        processWarning.classList.remove('hidden');
    }
}

// Process image function
async function processImage() {
    if (!selectedFile || !selectedLanguage || isProcessing) {
        alert('Please upload an image and select a language before processing.');
        return;
    }

    isProcessing = true;
    processBtn.classList.add('processing');
    processBtn.disabled = true;
    processWarning.classList.add('hidden');
    loader.style.display = 'inline-block';

    // Simulate processing delay
    await new Promise(resolve => setTimeout(resolve, 5000));
    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('language', selectedLanguage);
    // Send POST request to /result
    await fetch('/result', {
        method: 'POST',
        body: formData
    });
    redirectToResultWithPost(selectedFile, selectedLanguage);
    
}

function redirectToResultWithPost(file, language) {
    const form = document.createElement('form');
    form.action = '/result';
    // form.enctype = 'multipart/form-data';
    form.method = 'get';

    // File input
    // if (file) {
        // const fileInput = document.createElement('input');
        // fileInput.type = 'file';
        // fileInput.name = 'file';
        // You can't set a File object directly, so use FormData if you need to send the actual file.
        // Instead, append the file to FormData and submit via fetch, or use a hidden form and let the user select again.
        // For demo, we'll skip file here.
    // }

    // Language input
    // const langInput = document.createElement('input');
    // langInput.type = 'hidden';
    // langInput.name = 'language';
    // langInput.value = language;
    // form.appendChild(langInput);

    document.body.appendChild(form);
    form.submit();
}

// Event Listeners

// // Choose file button
// chooseFileBtn.addEventListener('click', (e) => {
//     e.stopPropagation();
//     fileInput.click();
// });

// Upload container click
uploadSection.addEventListener('click', () => {
    fileInput.click();
});

// File input change
fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
});

// Handle Enter key press for upload or process
document.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
        if (!selectedFile || !previewImage.src) {
            fileInput.click();
        } else {
            processBtn.click();
        }
    }
});

// Remove image button
removeImageBtn.addEventListener('click', removeImage);

// Language selection
document.querySelectorAll('input[name="language"]').forEach(radio => {
    radio.addEventListener('change', (e) => {
        selectedLanguage = e.target.value;
        updateProcessButton();
    });
});

// Process button
processBtn.addEventListener('click', processImage);

// Drag and drop functionality
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    uploadSection.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
    });
});

uploadSection.addEventListener('dragenter', () => {
    uploadSection.classList.add('drag-over');
});

uploadSection.addEventListener('dragover', () => {
    uploadSection.classList.add('drag-over');
});

uploadSection.addEventListener('dragleave', (e) => {
    // Only remove drag-over if we're actually leaving the upload section
    const rect = uploadSection.getBoundingClientRect();
    const x = e.clientX;
    const y = e.clientY;
    
    if (x < rect.left || x >= rect.right || y < rect.top || y >= rect.bottom) {
        uploadSection.classList.remove('drag-over');
    }
});

uploadSection.addEventListener('drop', (e) => {
    uploadSection.classList.remove('drag-over');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

// Initialize
updateProcessButton();