<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Radiation Fog Viewer</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/gif.js/0.2.0/gif.js"></script>
    <style>
        /* [Paste the exact CSS from your prompt here] */
        /* Change body background slightly to make yellow mist visible */
        body { font-family: 'Segoe UI', sans-serif; background: #e0e0e0; margin: 0; display: flex; flex-direction: column; align-items: center; padding: 10px; height: 100vh; box-sizing: border-box; }
        /* Keep other CSS the same */
    </style>
</head>
<body>

    <h2 id="page-title">Radiation Fog Forecast (Crossover Method)</h2>
    
    <div class="controls">
        <span id="run-info">Loading...</span> | 
        <span><strong>Yellow</strong> = Mist (1-3mi) | <strong>Purple</strong> = Dense Fog (&lt;0.5mi)</span>
        <button id="download-btn" onclick="generateGIF()">Download Animated GIF</button>
    </div>

    <div class="timeline-container" id="timeline"></div>

    <div id="map-container">
        <div id="loader">
            <div class="spinner"></div>
            <span>Loading Map...</span>
        </div>
        <img id="forecast-img" src="" alt="Map Display" onerror="this.style.display='none'">
    </div>

<script>
    // CONFIGURATION
    const totalHours = 18; // Matches the python script
    const imageDir = "images/";
    
    // --- DATE LOGIC ---
    // We want to find the most recent 23Z run.
    const now = new Date();
    
    // If it's before 23Z, show yesterday's 23Z run. 
    // If it's after 23Z, show today's 23Z run.
    if (now.getUTCHours() < 23) {
        now.setUTCDate(now.getUTCDate() - 1);
    }
    
    const yyyy = now.getUTCFullYear();
    const mm = String(now.getUTCMonth() + 1).padStart(2, '0');
    const dd = String(now.getUTCDate()).padStart(2, '0');
    const dateStr = `${yyyy}${mm}${dd}`;
    const runHour = '23'; // Fixed run time based on your Cron

    document.getElementById('run-info').textContent = `Run: ${dateStr} ${runHour}Z`;

    // --- ELEMENTS ---
    const timeline = document.getElementById('timeline');
    const imgElement = document.getElementById('forecast-img');
    const loader = document.getElementById('loader');
    const btn = document.getElementById('download-btn');
    let boxes = [];

    // --- IMAGE LOADING LOGIC ---
    function setImage(fhr) {
        const fhrStr = String(fhr).padStart(2, '0');
        // Filename matches Python script: fog_20231027_23z_f01.png
        const filename = `fog_${dateStr}_${runHour}z_f${fhrStr}.png`;
        
        loader.style.display = 'flex';
        imgElement.style.display = 'block';
        imgElement.src = imageDir + filename;

        boxes.forEach(b => b.classList.remove('active'));
        const activeBox = document.getElementById(`box-${fhr}`);
        if(activeBox) activeBox.classList.add('active');
    }

    imgElement.onload = function() { loader.style.display = 'none'; };

    // --- BUILD TIMELINE ---
    for (let i = 1; i <= totalHours; i++) {
        let box = document.createElement('div');
        box.className = 'time-box';
        box.textContent = `+${i}h`;
        box.id = `box-${i}`;
        box.addEventListener('mouseenter', () => setImage(i));
        box.addEventListener('click', () => setImage(i));
        timeline.appendChild(box);
        boxes.push(box);
    }

    setImage(1);

    // Keyboard Controls
    document.addEventListener('keydown', (e) => {
        let currentSrc = imgElement.src;
        let match = currentSrc.match(/_f(\d+)\.png/);
        if (!match) return;
        let currentFhr = parseInt(match[1]);

        if (e.key === "ArrowRight") {
            let next = currentFhr + 1;
            if (next <= totalHours) setImage(next);
        } else if (e.key === "ArrowLeft") {
            let prev = currentFhr - 1;
            if (prev >= 1) setImage(prev);
        }
    });

    // --- GIF GENERATION LOGIC ---
    // [Keep the exact GIF generation logic from your previous file]
    async function generateGIF() {
       // ... (Use the same code as your previous HTML)
       // Just ensure filenames inside this function match the new `fog_...` convention
        const gif = new GIF({ workers: 2, quality: 10, width: imgElement.naturalWidth, height: imgElement.naturalHeight });
        for (let i = 1; i <= totalHours; i++) {
            const fhrStr = String(i).padStart(2, '0');
            const filename = `fog_${dateStr}_${runHour}z_f${fhrStr}.png`; // Updated filename
            const url = imageDir + filename;
            // ... rest of logic
        }
       // ...
    }
</script>
</body>
</html>
