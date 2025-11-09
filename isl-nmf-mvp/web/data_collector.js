// ============================================================================
// Data Collection Helper
// Add this to app.js or use in browser console for collecting training data
// ============================================================================

// Data collection state
let isCollecting = false;
let collectedSamples = [];
let currentCollectionLabel = '';

// Add UI for data collection
function addDataCollectionUI() {
  const collectDiv = document.createElement('div');
  collectDiv.id = 'dataCollect';
  collectDiv.innerHTML = `
    <div style="position: fixed; top: 10px; right: 10px; background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.2); z-index: 1000;">
      <h3 style="margin: 0 0 10px 0; font-size: 14px;">Data Collection</h3>
      <input type="text" id="labelInput" placeholder="Enter label (e.g., question)" style="width: 200px; padding: 5px; margin-bottom: 8px;">
      <br>
      <button id="collectBtn" style="padding: 8px 16px; margin-right: 5px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer;">Start Collecting</button>
      <button id="downloadBtn" style="padding: 8px 16px; background: #28a745; color: white; border: none; border-radius: 4px; cursor: pointer;">Download CSV</button>
      <div id="collectStatus" style="margin-top: 8px; font-size: 12px; color: #666;">
        Samples: <span id="sampleCount">0</span>
      </div>
    </div>
  `;
  document.body.appendChild(collectDiv);

  // Add event listeners
  document.getElementById('collectBtn').addEventListener('click', toggleCollection);
  document.getElementById('downloadBtn').addEventListener('click', downloadCSV);
}

function toggleCollection() {
  const btn = document.getElementById('collectBtn');
  const labelInput = document.getElementById('labelInput');
  
  if (!isCollecting) {
    currentCollectionLabel = labelInput.value.trim();
    if (!currentCollectionLabel) {
      alert('Please enter a label first!');
      return;
    }
    isCollecting = true;
    btn.textContent = 'Stop Collecting';
    btn.style.background = '#dc3545';
    console.log(`Started collecting samples for: ${currentCollectionLabel}`);
  } else {
    isCollecting = false;
    btn.textContent = 'Start Collecting';
    btn.style.background = '#007bff';
    console.log('Stopped collecting');
  }
}

function collectSample() {
  if (!isCollecting) return;
  
  // Collect current features with label
  const sample = {
    eyebrow_raise: currentFeatures.eyebrowRaise,
    mouth_open: currentFeatures.mouthOpen,
    head_tilt: currentFeatures.headTilt,
    head_nod: currentFeatures.headNod,
    head_shake: currentFeatures.headShake,
    shoulder_lean: currentFeatures.shoulderLean,
    label: currentCollectionLabel
  };
  
  collectedSamples.push(sample);
  document.getElementById('sampleCount').textContent = collectedSamples.length;
  
  // Log every 10 samples
  if (collectedSamples.length % 10 === 0) {
    console.log(`Collected ${collectedSamples.length} samples`);
  }
}

function downloadCSV() {
  if (collectedSamples.length === 0) {
    alert('No samples collected yet!');
    return;
  }
  
  // Convert to CSV
  const headers = ['eyebrow_raise', 'mouth_open', 'head_tilt', 'head_nod', 'head_shake', 'shoulder_lean', 'label'];
  let csv = headers.join(',') + '\n';
  
  collectedSamples.forEach(sample => {
    const row = headers.map(h => sample[h]).join(',');
    csv += row + '\n';
  });
  
  // Download
  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `nmf_data_${Date.now()}.csv`;
  a.click();
  URL.revokeObjectURL(url);
  
  console.log(`Downloaded ${collectedSamples.length} samples`);
  alert(`Downloaded ${collectedSamples.length} samples!`);
}

// ============================================================================
// Instructions for use:
// ============================================================================

/*
1. Add this code to your app.js file, or paste in browser console

2. Call addDataCollectionUI() after page loads:
   window.addEventListener('load', addDataCollectionUI);

3. Add collectSample() call to processFrame():
   function processFrame() {
     // ... existing code ...
     collectSample(); // Add this line
   }

4. Usage:
   - Enter a label (e.g., "question", "surprise", "neutral")
   - Click "Start Collecting"
   - Perform the expression/gesture
   - Hold for a few seconds to collect multiple samples
   - Click "Stop Collecting"
   - Repeat for different labels
   - Click "Download CSV" when done

5. Merge downloaded CSVs into dataset/nmf_training.csv

6. Run train_classifier.py to train on new data

Tips for good data:
- Collect 20-50 samples per expression
- Include variations (slight, moderate, exaggerated)
- Collect "neutral" baseline samples
- Different lighting conditions
- Different head positions
- Different people if possible
*/

console.log('Data collection helper loaded. Call addDataCollectionUI() to start.');
