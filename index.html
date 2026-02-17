<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Crossover Fog Forecast</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/gif.js/0.2.0/gif.js"></script>
    <style>
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: #f4f4f9; 
            margin: 0; 
            display: flex; 
            flex-direction: column; 
            align-items: center; 
            padding: 10px; 
            height: 100vh; 
            box-sizing: border-box; 
        }
        h2 { margin: 5px 0; color: #333; font-size: 1.2rem; }
        
        /* Control Bar */
        .controls { 
            margin-bottom: 8px; 
            font-size: 13px; 
            color: #555; 
            display: flex; 
            flex-wrap: wrap;
            justify-content: center;
            align-items: center; 
            gap: 10px; 
        }
        
        button {
            padding: 6px 12px;
            border: 1px solid #ccc;
            background: white;
            cursor: pointer;
            border-radius: 4px;
            font-size: 12px;
        }
        button:hover { background: #eee; }
        button#download-btn {
            background-color: #673AB7; 
            color: white; border: none; font-weight: bold;
        }
        button#download-btn:hover { background-color: #512DA8; }
        button#download-btn:disabled { background-color: #ccc; cursor: not-allowed; }

        /* Timeline */
        .timeline-container {
            display: grid; grid-template-columns: repeat(18, 1fr); 
            width: 100%; max-width: 800px; gap: 1px; margin-bottom: 5px; 
            background: #fff; padding: 3px; border-radius: 5px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .time-box {
            background: #e9ecef; border: 1px solid #ced4da; 
            text-align: center; font-size: 10px; padding: 4px 0; 
            cursor: pointer; user-select: none;
        }
        .time-box:hover { background: #d1c4e9; }
        .time-box.active { background: #673AB7; color: white; font-weight: bold; }

        /* Map Area */
        #map-container {
            flex-grow: 1;
            width: 100%; max-width: 1200px;
            background: #fff; border: 1px solid #ccc; 
            display: flex; align-items: center; justify-content: center; 
            padding: 5px; position: relative;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        #forecast-img { 
            max-height: 100%; max-width: 100%; 
            object-fit: contain; 
            display: block; 
        }
        
        /* Loaders & Errors */
        #loader, #error-msg { 
            position: absolute; 
            display: flex; 
            flex-direction: column; 
            align-items: center; 
            justify-content: center;
            background: rgba(255,255,255,0.9);
            padding: 20px;
            border-radius: 8px;
        }
        #error-msg { display: none; color: red; text-align: center; }
        .spinner { 
            border: 4px solid #f3f3f3; 
            border-top: 4px solid #673AB7; 
            border-radius: 50%; 
            width: 40px; height: 40px; 
            animation: spin 1s linear infinite; 
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    </style>
</head>
<body>

    <h2 id="page-title">Crossover Fog Forecast</h2>
    
    <div class="controls">
        <button onclick="toggleView('forecast')" id="btn-forecast" style="font-weight:bold; border-color:#673AB7;">Forecast Loop</button>
        <button onclick="toggleView('analysis')" id="btn-analysis">View Input Analysis</button>
        <button id="download-btn" onclick="generateGIF()">Download GIF</button>
    </div>
    
    <div class="controls">
        <span id="run-info">Initializing...</span> | 
        <span><span style="color:#FFEB3B; background:#555; padding:0 4px; border-radius:3px;">Yellow = Mist</span> <span style="color:#9C27B0; font-weight:bold;">Purple = Dense Fog</span></span>
    </div>

    <div class="timeline-container" id="timeline"></div>

    <div id="map-container">
        <div id="loader">
            <div class="spinner"></div>
            <br><span id="loader-text">Finding latest run...</span>
        </div>
        <div id="error-msg">
            <strong>Map Not Found</strong><br>
            <span id="error-details">Waiting for data...</span>
        </div>
        <img id="forecast-img" src="" alt="Map Display">
    </div>

<script>
    // CONFIGURATION
    const totalHours = 18; 
    const imageDir = "images/";
    
    // --- ROBUST DATE LOGIC ---
    // We try to find the "Latest" 23Z run. 
    // If it's 20:00 UTC today, the latest run was Yesterday 23Z.
    // If it's 23:30 UTC today, the latest run is Today 23Z.
    
    const now = new Date();
    const currentUtcHour = now.getUTCHours();
    
    let runDate = new Date(now);
    // If it's before 23:00 UTC, we MUST show yesterday's data
    if (currentUtcHour < 23) {
        runDate.setUTCDate(runDate.getUTCDate() - 1);
    }
    
    const yyyy = runDate.getUTCFullYear();
    const mm = String(runDate.getUTCMonth() + 1).padStart(2, '0');
    const dd = String(runDate.getUTCDate()).padStart(2, '0');
    const dateStr = `${yyyy}${mm}${dd}`;
    const runHour = '23'; 

    document.getElementById('run-info').textContent = `Run: ${dateStr} ${runHour}Z`;

    // --- VIEW LOGIC ---
    let currentView = 'forecast';
    const timeline = document.getElementById('timeline');
    const imgElement = document.getElementById('forecast-img');
    const loader = document.getElementById('loader');
    const errorMsg = document.getElementById('error-msg');
    const btn = document.getElementById('download-btn');
    let boxes = [];

    function toggleView(view) {
        currentView = view;
        errorMsg.style.display = 'none'; // Clear errors on toggle
        
        if (view === 'analysis') {
            timeline.style.display = 'none';
            // Force a random query param to bust cache (?v=...)
            loadImage(`crossover_analysis.png?v=${new Date().getTime()}`);
            document.getElementById('btn-analysis').style.fontWeight = 'bold';
            document.getElementById('btn-analysis').style.borderColor = '#673AB7';
            document.getElementById('btn-forecast').style.fontWeight = 'normal';
            document.getElementById('btn-forecast').style.borderColor = '#ccc';
            btn.style.display = 'none'; // Hide GIF button for static map
        } else {
            timeline.style.display = 'grid';
            setImage(1); 
            document.getElementById('btn-analysis').style.fontWeight = 'normal';
            document.getElementById('btn-analysis').style.borderColor = '#ccc';
            document.getElementById('btn-forecast').style.fontWeight = 'bold';
            document.getElementById('btn-forecast').style.borderColor = '#673AB7';
            btn.style.display = 'block';
        }
    }

    // --- IMAGE LOADER ---
    function setImage(fhr) {
        const fhrStr = String(fhr).padStart(2, '0');
        // Add cache buster ?v=... to force browser to load new image if it changed
        const filename = `fog_${dateStr}_${runHour}z_f${fhrStr}.png?v=${new Date().getTime()}`;
        loadImage(filename);

        // Highlight timeline
        boxes.forEach(b => b.classList.remove('active'));
        const activeBox = document.getElementById(`box-${fhr}`);
        if(activeBox) activeBox.classList.add('active');
    }

    function loadImage(filename) {
        loader.style.display = 'flex';
        imgElement.style.display = 'none';
        errorMsg.style.display = 'none';
        
        imgElement.src = imageDir + filename;
    }

    imgElement.onload = function() { 
        loader.style.display = 'none';
        imgElement.style.display = 'block';
    };

    imgElement.onerror = function() {
        loader.style.display = 'none';
        imgElement.style.display = 'none';
        errorMsg.style.display = 'flex';
        document.getElementById('error-details').textContent = `File not found:\n${this.src.split('/').pop().split('?')[0]}`;
    };

    // --- BUILD TIMELINE ---
    for (let i = 1; i <= totalHours; i++) {
        let box = document.createElement('div');
        box.className = 'time-box';
        box.textContent = `+${i}h`;
        box.id = `box-${i}`;
        box.addEventListener('mouseenter', () => { if(currentView === 'forecast') setImage(i); });
        box.addEventListener('click', () => { if(currentView === 'forecast') setImage(i); });
        timeline.appendChild(box);
        boxes.push(box);
    }

    // Initialize
    setImage(1);

    // --- GIF GENERATION ---
    async function generateGIF() {
        btn.disabled = true;
        btn.textContent = "Processing...";
        try {
            const workerResponse = await fetch('https://cdnjs.cloudflare.com/ajax/libs/gif.js/0.2.0/gif.worker.js');
            const workerBlob = await workerResponse.blob();
            const workerUrl = URL.createObjectURL(workerBlob);

            const gif = new GIF({ workers: 2, quality: 10, workerScript: workerUrl, width: imgElement.naturalWidth, height: imgElement.naturalHeight });

            for (let i = 1; i <= totalHours; i++) {
                const fhrStr = String(i).padStart(2, '0');
                const filename = `fog_${dateStr}_${runHour}z_f${fhrStr}.png`;
                const img = new Image();
                img.crossOrigin = "Anonymous";
                await new Promise((resolve) => {
                    img.onload = () => { gif.addFrame(img, {delay: 250}); resolve(); };
                    img.onerror = () => { resolve(); }; // Skip missing frames
                    img.src = imageDir + filename;
                });
            }
            gif.on('finished', function(blob) {
                const link = document.createElement('a');
                link.href = URL.createObjectURL(blob);
                link.download = `fog_loop_${dateStr}.gif`;
                link.click();
                btn.textContent = "Download GIF";
                btn.disabled = false;
            });
            gif.render();
        } catch (e) {
            console.error(e);
            btn.textContent = "Error";
            alert("GIF Error: " + e.message);
        }
    }
</script>
</body>
</html>
