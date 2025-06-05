document.addEventListener('DOMContentLoaded', function() {
    // Funzione principale che carica e processa i dati
    async function loadDashboardData() {
        try {
            const response = await fetch('data.json');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            
            populateStatus(data.projectInfo);
            populateKPIs(data.kpis);
            populateGantt(data.roadmapTasks, data.projectInfo);
            populateMilestones(data.milestones);
            populateParadigms(data.paradigms);

            // Calcola e aggiorna il progresso totale dopo aver popolato il Gantt
            calculateAndUpdateOverallProgress(data.roadmapTasks, data.projectInfo);

        } catch (error) {
            console.error("Could not load dashboard data:", error);
            // Mostra un errore all'utente
            document.body.innerHTML = '<div style="text-align: center; padding-top: 50px; color: #e94560;"><h1>Error</h1><p>Could not load project data. Please check data.json and the console for more information.</p></div>';
        }
    }
    
    function populateStatus(info) {
        const container = document.getElementById('status-badge-container');
        const statusClass = info.status.toLowerCase().replace(/ /g, '-');
        container.innerHTML = `
            <div class="status-badge ${statusClass}">
                <i class="fas fa-info-circle"></i> STATUS: ${info.status.toUpperCase()}
            </div>
        `;
    }

    function populateKPIs(kpis) {
        const grid = document.getElementById('kpi-grid');
        grid.innerHTML = ''; // Pulisce il contenitore
        kpis.forEach(kpi => {
            const card = document.createElement('div');
            card.className = `kpi-card ${kpi.highlight ? 'highlight' : ''}`;
            
            let content;
            if (kpi.id === 'overall-progress') {
                content = `
                    <h3><i class="${kpi.icon}"></i> ${kpi.title}</h3>
                    <div class="progress-circle" id="overall-progress-circle">
                        <span id="overall-progress-text">0%</span>
                    </div>
                    <p>${kpi.description}</p>
                `;
            } else if (kpi.id === 'current-focus') {
                 content = `
                    <h3><i class="${kpi.icon}"></i> ${kpi.title}</h3>
                    <div class="current-focus-content">
                        <p class="focus-title">${kpi.value}</p>
                        <div class="progress-bar-container">
                            <div class="progress-bar" style="width: ${kpi.progress}%;"></div>
                        </div>
                    </div>
                    <p>${kpi.description}</p>
                `;
            } else {
                content = `
                    <h3><i class="${kpi.icon}"></i> ${kpi.title}</h3>
                    <div class="kpi-value">${kpi.value}${kpi.unit || ''}</div>
                    <p>${kpi.description}</p>
                `;
            }
            card.innerHTML = content;
            grid.appendChild(card);
        });
    }

    function populateGantt(tasks, info) {
        const ganttChart = document.getElementById('gantt-chart');
        ganttChart.innerHTML = '';
        const today = new Date(info.today);
        const timelineStart = new Date(info.timelineStart);
        const timelineEnd = new Date(info.timelineEnd);
        const totalDuration = timelineEnd - timelineStart;
        
        let currentSection = '';
        tasks.forEach(task => {
            if (task.section !== currentSection) {
                currentSection = task.section;
                const sectionHeader = document.createElement('div');
                sectionHeader.className = 'gantt-section-header';
                sectionHeader.textContent = currentSection;
                ganttChart.appendChild(sectionHeader);
            }

            const taskStart = new Date(task.start);
            const taskEnd = new Date(task.end);

            const left = ((taskStart - timelineStart) / totalDuration) * 100;
            const width = ((taskEnd - taskStart) / totalDuration) * 100;

            const progress = calculateProgress(today, taskStart, taskEnd);
            
            let statusClass = 'pending';
            if (progress > 0 && progress < 100) statusClass = 'in-progress';
            if (progress === 100) statusClass = 'completed';
            
            const rowHTML = `
                <div class="gantt-row">
                    <div class="gantt-label">${task.name}</div>
                    <div class="gantt-bar-container">
                        <div class="gantt-bar ${statusClass}" style="left: ${left}%; width: ${width}%;">
                            <div class="gantt-progress" style="width: ${progress}%;"></div>
                        </div>
                    </div>
                </div>
            `;
            ganttChart.insertAdjacentHTML('beforeend', rowHTML);
        });
    }

    function populateMilestones(milestones) {
        const list = document.getElementById('milestone-list');
        list.innerHTML = '';
        milestones.forEach(m => {
            const statusClass = m.status.toLowerCase();
            const iconClass = statusClass === 'completed' ? 'fa-check-circle' : 'fa-hourglass-half';
            const item = document.createElement('li');
            item.className = statusClass;
            item.innerHTML = `<i class="fas ${iconClass}"></i> <strong>${m.quarter}:</strong> ${m.description} <span>(${m.status})</span>`;
            list.appendChild(item);
        });
    }

    function populateParadigms(paradigms) {
        const container = document.getElementById('paradigms-container');
        container.innerHTML = '';
        paradigms.forEach(p => {
            const paradigmHTML = `
                <div class="paradigm">
                    <h4>${p.title}</h4>
                    <p>${p.description}</p>
                </div>
            `;
            container.insertAdjacentHTML('beforeend', paradigmHTML);
        });
    }
    
    function calculateProgress(today, start, end) {
        if (today >= end) return 100;
        if (today < start) return 0;
        const taskDuration = end - start;
        const elapsed = today - start;
        return (elapsed / taskDuration) * 100;
    }

    function calculateAndUpdateOverallProgress(tasks, info) {
        let totalProgress = 0;
        const today = new Date(info.today);
        tasks.forEach(task => {
            const taskStart = new Date(task.start);
            const taskEnd = new Date(task.end);
            totalProgress += calculateProgress(today, taskStart, taskEnd);
        });
        const overallPercentage = Math.round(totalProgress / tasks.length);

        const progressCircle = document.getElementById('overall-progress-circle');
        const progressText = document.getElementById('overall-progress-text');
        
        if(progressCircle && progressText) {
            progressText.textContent = `${overallPercentage}%`;
            progressCircle.style.background = `conic-gradient(#00e0ff ${overallPercentage * 3.6}deg, #2a2a3e ${overallPercentage * 3.6}deg)`;
        }
    }

    // Avvia il caricamento
    loadDashboardData();
});