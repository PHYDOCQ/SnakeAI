/**
 * Advanced Snake AI Web Interface JavaScript
 * Handles real-time communication with Flask backend and UI updates
 */

class SnakeAIInterface {
    constructor() {
        this.gameRunning = false;
        this.trainingActive = false;
        this.currentAgent = 'rainbow';
        this.autoPlay = false;
        this.gameStepInterval = null;
        this.statusUpdateInterval = null;
        this.metricsUpdateInterval = null;
        
        // API endpoints
        this.apiBase = '/api';
        
        // UI elements
        this.elements = {};
        
        // Game state
        this.gameState = null;
        this.aiAnalysis = null;
        
        // Initialize when DOM is ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.initialize());
        } else {
            this.initialize();
        }
    }
    
    initialize() {
        console.log('Initializing Snake AI Interface...');
        
        // Cache DOM elements
        this.cacheElements();
        
        // Bind event listeners
        this.bindEvents();
        
        // Start status updates
        this.startStatusUpdates();
        
        // Initialize visualization options
        this.initializeVisualizationOptions();
        
        // Load initial data
        this.loadInitialData();
        
        console.log('Snake AI Interface initialized successfully');
    }
    
    cacheElements() {
        // Game controls
        this.elements.startGameBtn = document.getElementById('startGame');
        this.elements.stopGameBtn = document.getElementById('stopGame');
        this.elements.stepGameBtn = document.getElementById('stepGame');
        this.elements.autoPlayBtn = document.getElementById('autoPlay');
        
        // Training controls
        this.elements.startTrainingBtn = document.getElementById('startTraining');
        this.elements.stopTrainingBtn = document.getElementById('stopTraining');
        
        // Agent selection
        this.elements.agentButtons = document.querySelectorAll('.agent-btn');
        
        // Status indicators
        this.elements.systemStatus = document.getElementById('systemStatus');
        this.elements.trainingStatus = document.getElementById('trainingStatus');
        this.elements.gameStatus = document.getElementById('gameStatus');
        
        // Statistics
        this.elements.episodeCount = document.getElementById('episodeCount');
        this.elements.bestScore = document.getElementById('bestScore');
        this.elements.currentScore = document.getElementById('currentScore');
        this.elements.recentAverage = document.getElementById('recentAverage');
        
        // Progress bars
        this.elements.curriculumProgress = document.getElementById('curriculumProgress');
        this.elements.curriculumStage = document.getElementById('curriculumStage');
        
        // AI Analysis
        this.elements.actionTaken = document.getElementById('actionTaken');
        this.elements.confidence = document.getElementById('confidence');
        this.elements.decisionQuality = document.getElementById('decisionQuality');
        this.elements.aiExplanation = document.getElementById('aiExplanation');
        this.elements.strategyType = document.getElementById('strategyType');
        this.elements.riskLevel = document.getElementById('riskLevel');
        
        // Probability bars
        this.elements.probBars = {
            up: document.getElementById('probUp'),
            right: document.getElementById('probRight'),
            down: document.getElementById('probDown'),
            left: document.getElementById('probLeft')
        };
        
        // Game canvas
        this.elements.gameCanvas = document.getElementById('gameCanvas');
        this.elements.canvasContainer = document.getElementById('gameCanvasContainer');
        
        // Visualization options
        this.elements.vizOptions = document.querySelectorAll('.viz-option');
        
        // Metrics
        this.elements.metricsContainer = document.getElementById('metricsContainer');
        this.elements.metricsTabs = document.querySelectorAll('.metrics-tab');
        
        // Error display
        this.elements.errorContainer = document.getElementById('errorContainer');
    }
    
    bindEvents() {
        // Game control events
        if (this.elements.startGameBtn) {
            this.elements.startGameBtn.addEventListener('click', () => this.startGame());
        }
        
        if (this.elements.stopGameBtn) {
            this.elements.stopGameBtn.addEventListener('click', () => this.stopGame());
        }
        
        if (this.elements.stepGameBtn) {
            this.elements.stepGameBtn.addEventListener('click', () => this.stepGame());
        }
        
        if (this.elements.autoPlayBtn) {
            this.elements.autoPlayBtn.addEventListener('click', () => this.toggleAutoPlay());
        }
        
        // Training control events
        if (this.elements.startTrainingBtn) {
            this.elements.startTrainingBtn.addEventListener('click', () => this.startTraining());
        }
        
        if (this.elements.stopTrainingBtn) {
            this.elements.stopTrainingBtn.addEventListener('click', () => this.stopTraining());
        }
        
        // Agent selection events
        this.elements.agentButtons.forEach(btn => {
            btn.addEventListener('click', (e) => this.selectAgent(e.target.dataset.agent));
        });
        
        // Visualization option events
        this.elements.vizOptions.forEach(option => {
            option.addEventListener('click', () => this.toggleVisualizationOption(option));
        });
        
        // Metrics tab events
        this.elements.metricsTabs.forEach(tab => {
            tab.addEventListener('click', (e) => this.switchMetricsTab(e.target.dataset.tab));
        });
        
        // Keyboard controls
        document.addEventListener('keydown', (e) => this.handleKeyPress(e));
    }
    
    async startGame() {
        try {
            this.showLoading('Starting game...');
            
            const response = await this.apiCall('/game/start', 'POST', {
                agent: this.currentAgent
            });
            
            if (response.success) {
                this.gameRunning = true;
                this.updateGameControls();
                this.showSuccess('Game started successfully');
                
                // Start auto-play if enabled
                if (this.autoPlay) {
                    this.startAutoPlay();
                }
            } else {
                this.showError(response.message || 'Failed to start game');
            }
        } catch (error) {
            this.showError(`Error starting game: ${error.message}`);
        } finally {
            this.hideLoading();
        }
    }
    
    async stopGame() {
        try {
            this.showLoading('Stopping game...');
            
            const response = await this.apiCall('/game/stop', 'POST');
            
            if (response.success) {
                this.gameRunning = false;
                this.stopAutoPlay();
                this.updateGameControls();
                this.showSuccess('Game stopped');
            } else {
                this.showError(response.message || 'Failed to stop game');
            }
        } catch (error) {
            this.showError(`Error stopping game: ${error.message}`);
        } finally {
            this.hideLoading();
        }
    }
    
    async stepGame() {
        if (!this.gameRunning) return;
        
        try {
            const response = await this.apiCall('/game/step', 'POST');
            
            if (response.error) {
                this.showError(response.error);
                return;
            }
            
            // Update game state
            this.gameState = response.game_state;
            this.aiAnalysis = response.ai_analysis;
            
            // Update UI
            this.updateGameDisplay();
            this.updateAIAnalysis();
            
            // Check if game ended
            if (response.done && this.autoPlay) {
                // Brief pause before restarting
                setTimeout(() => {
                    if (this.autoPlay && this.gameRunning) {
                        this.startGame(); // Restart the game
                    }
                }, 1000);
            }
            
        } catch (error) {
            this.showError(`Error in game step: ${error.message}`);
        }
    }
    
    toggleAutoPlay() {
        this.autoPlay = !this.autoPlay;
        
        if (this.autoPlay && this.gameRunning) {
            this.startAutoPlay();
        } else {
            this.stopAutoPlay();
        }
        
        this.updateGameControls();
    }
    
    startAutoPlay() {
        if (this.gameStepInterval) {
            clearInterval(this.gameStepInterval);
        }
        
        this.gameStepInterval = setInterval(() => {
            if (this.gameRunning && this.autoPlay) {
                this.stepGame();
            }
        }, 200); // 5 FPS
    }
    
    stopAutoPlay() {
        if (this.gameStepInterval) {
            clearInterval(this.gameStepInterval);
            this.gameStepInterval = null;
        }
    }
    
    async startTraining() {
        try {
            this.showLoading('Starting training...');
            
            const response = await this.apiCall('/training/start', 'POST');
            
            if (response.success) {
                this.trainingActive = true;
                this.updateTrainingControls();
                this.showSuccess('Training started successfully');
                
                // Start metrics updates
                this.startMetricsUpdates();
            } else {
                this.showError(response.message || 'Failed to start training');
            }
        } catch (error) {
            this.showError(`Error starting training: ${error.message}`);
        } finally {
            this.hideLoading();
        }
    }
    
    async stopTraining() {
        try {
            this.showLoading('Stopping training...');
            
            const response = await this.apiCall('/training/stop', 'POST');
            
            if (response.success) {
                this.trainingActive = false;
                this.updateTrainingControls();
                this.showSuccess('Training stopped');
                
                // Stop metrics updates
                this.stopMetricsUpdates();
            } else {
                this.showError(response.message || 'Failed to stop training');
            }
        } catch (error) {
            this.showError(`Error stopping training: ${error.message}`);
        } finally {
            this.hideLoading();
        }
    }
    
    selectAgent(agentType) {
        this.currentAgent = agentType;
        
        // Update UI
        this.elements.agentButtons.forEach(btn => {
            btn.classList.toggle('active', btn.dataset.agent === agentType);
        });
        
        this.showInfo(`Selected ${agentType} agent`);
    }
    
    async updateGameVisualization() {
        if (!this.gameRunning) return;
        
        try {
            const response = await this.apiCall('/visualization/image');
            
            if (response.error) {
                console.warn('Failed to get visualization:', response.error);
                return;
            }
            
            // Update canvas with new image
            if (response.image && this.elements.gameCanvas) {
                this.elements.gameCanvas.src = response.image;
            }
            
        } catch (error) {
            console.warn('Error updating visualization:', error);
        }
    }
    
    updateGameDisplay() {
        if (!this.gameState) return;
        
        // Update game statistics
        if (this.elements.currentScore) {
            this.elements.currentScore.textContent = this.gameState.score;
        }
        
        // Update canvas overlay info
        const overlay = document.querySelector('.canvas-overlay');
        if (overlay) {
            overlay.innerHTML = `
                <div>Score: ${this.gameState.score}</div>
                <div>Length: ${this.gameState.snake_length}</div>
                <div>Steps: ${this.gameState.steps}</div>
            `;
        }
        
        // Update visualization
        this.updateGameVisualization();
    }
    
    updateAIAnalysis() {
        if (!this.aiAnalysis) return;
        
        // Update action taken
        if (this.elements.actionTaken) {
            this.elements.actionTaken.textContent = this.aiAnalysis.action_name || 'Unknown';
        }
        
        // Update confidence
        if (this.elements.confidence) {
            const confidence = (this.aiAnalysis.confidence * 100).toFixed(1);
            this.elements.confidence.textContent = `${confidence}%`;
        }
        
        // Update decision quality
        if (this.elements.decisionQuality) {
            const quality = (this.aiAnalysis.decision_quality * 100).toFixed(1);
            this.elements.decisionQuality.textContent = `${quality}%`;
        }
        
        // Update explanation
        if (this.elements.aiExplanation) {
            this.elements.aiExplanation.textContent = this.aiAnalysis.explanation || 'No explanation available';
        }
        
        // Update strategy and risk
        if (this.elements.strategyType) {
            this.elements.strategyType.textContent = this.aiAnalysis.strategy_type || 'Unknown';
        }
        
        if (this.elements.riskLevel) {
            this.elements.riskLevel.textContent = this.aiAnalysis.risk_level || 'Unknown';
        }
        
        // Update probability bars
        if (this.aiAnalysis.action_probabilities) {
            this.updateProbabilityBars(this.aiAnalysis.action_probabilities);
        }
    }
    
    updateProbabilityBars(probabilities) {
        const actions = ['up', 'right', 'down', 'left'];
        
        actions.forEach((action, index) => {
            if (this.elements.probBars[action] && probabilities[index] !== undefined) {
                const prob = probabilities[index];
                const percentage = (prob * 100).toFixed(1);
                
                // Update bar fill
                const fill = this.elements.probBars[action].querySelector('.prob-bar-value');
                if (fill) {
                    fill.style.width = `${percentage}%`;
                }
                
                // Update text
                const text = this.elements.probBars[action].querySelector('.prob-bar-text');
                if (text) {
                    text.textContent = `${percentage}%`;
                }
            }
        });
    }
    
    updateGameControls() {
        // Update button states
        if (this.elements.startGameBtn) {
            this.elements.startGameBtn.disabled = this.gameRunning;
        }
        
        if (this.elements.stopGameBtn) {
            this.elements.stopGameBtn.disabled = !this.gameRunning;
        }
        
        if (this.elements.stepGameBtn) {
            this.elements.stepGameBtn.disabled = !this.gameRunning || this.autoPlay;
        }
        
        if (this.elements.autoPlayBtn) {
            this.elements.autoPlayBtn.textContent = this.autoPlay ? 'Stop Auto' : 'Auto Play';
            this.elements.autoPlayBtn.classList.toggle('active', this.autoPlay);
        }
        
        // Update game status indicator
        if (this.elements.gameStatus) {
            this.elements.gameStatus.textContent = this.gameRunning ? 'Running' : 'Stopped';
            this.elements.gameStatus.className = this.gameRunning ? 'status-active' : 'status-inactive';
        }
    }
    
    updateTrainingControls() {
        // Update button states
        if (this.elements.startTrainingBtn) {
            this.elements.startTrainingBtn.disabled = this.trainingActive;
        }
        
        if (this.elements.stopTrainingBtn) {
            this.elements.stopTrainingBtn.disabled = !this.trainingActive;
        }
        
        // Update training status indicator
        if (this.elements.trainingStatus) {
            this.elements.trainingStatus.textContent = this.trainingActive ? 'Active' : 'Stopped';
            this.elements.trainingStatus.className = this.trainingActive ? 'status-active' : 'status-inactive';
        }
    }
    
    async loadStatus() {
        try {
            const status = await this.apiCall('/status');
            
            if (status.error) {
                this.showError('Failed to load system status');
                return;
            }
            
            // Update system state
            this.gameRunning = status.game_running;
            this.trainingActive = status.training_active;
            this.currentAgent = status.current_agent;
            
            // Update UI elements
            this.updateSystemStatus(status);
            this.updateStatistics(status);
            this.updateGameControls();
            this.updateTrainingControls();
            
            // Update agent selection
            this.elements.agentButtons.forEach(btn => {
                btn.classList.toggle('active', btn.dataset.agent === this.currentAgent);
            });
            
        } catch (error) {
            this.showError(`Error loading status: ${error.message}`);
        }
    }
    
    updateSystemStatus(status) {
        // Update system status indicator
        if (this.elements.systemStatus) {
            const statusElement = this.elements.systemStatus.querySelector('.status-dot');
            if (statusElement) {
                statusElement.className = 'status-dot ' + (status.system_online ? 'online' : 'offline');
            }
        }
        
        // Update curriculum progress
        if (this.elements.curriculumStage && status.curriculum_stage) {
            this.elements.curriculumStage.textContent = status.curriculum_stage;
        }
        
        if (this.elements.curriculumProgress && status.curriculum_progress) {
            const progress = status.curriculum_progress.current_stage?.progress || 0;
            this.elements.curriculumProgress.style.width = `${progress * 100}%`;
        }
    }
    
    updateStatistics(status) {
        // Update episode count
        if (this.elements.episodeCount) {
            this.elements.episodeCount.textContent = status.episode_count || 0;
        }
        
        // Update best score
        if (this.elements.bestScore) {
            this.elements.bestScore.textContent = status.best_score || 0;
        }
        
        // Update recent average
        if (this.elements.recentAverage) {
            const avg = status.recent_average ? status.recent_average.toFixed(2) : '0.00';
            this.elements.recentAverage.textContent = avg;
        }
    }
    
    startStatusUpdates() {
        // Load initial status
        this.loadStatus();
        
        // Set up periodic updates
        this.statusUpdateInterval = setInterval(() => {
            this.loadStatus();
        }, 2000); // Update every 2 seconds
    }
    
    stopStatusUpdates() {
        if (this.statusUpdateInterval) {
            clearInterval(this.statusUpdateInterval);
            this.statusUpdateInterval = null;
        }
    }
    
    async loadMetrics() {
        try {
            const metrics = await this.apiCall('/training/metrics');
            
            if (metrics.error) {
                console.warn('Failed to load metrics:', metrics.error);
                return;
            }
            
            this.updateMetricsDisplay(metrics);
            
        } catch (error) {
            console.warn('Error loading metrics:', error);
        }
    }
    
    updateMetricsDisplay(metrics) {
        // Update training metrics images
        const metricsImage = document.getElementById('trainingMetricsImage');
        if (metricsImage && metrics.visualizations?.training_metrics) {
            metricsImage.src = metrics.visualizations.training_metrics;
        }
        
        const curriculumImage = document.getElementById('curriculumProgressImage');
        if (curriculumImage && metrics.visualizations?.curriculum_progress) {
            curriculumImage.src = metrics.visualizations.curriculum_progress;
        }
        
        // Update agent performance comparison
        this.updateAgentComparison(metrics.agent_performance);
    }
    
    updateAgentComparison(agentPerformance) {
        const container = document.getElementById('agentComparison');
        if (!container || !agentPerformance) return;
        
        container.innerHTML = '';
        
        Object.entries(agentPerformance).forEach(([agentName, performance]) => {
            const scores = performance.scores || [];
            const avgScore = scores.length > 0 ? (scores.reduce((a, b) => a + b, 0) / scores.length).toFixed(2) : '0.00';
            
            const item = document.createElement('div');
            item.className = 'comparison-item';
            item.innerHTML = `
                <span class="agent-name">${agentName}</span>
                <span class="agent-score">${avgScore}</span>
            `;
            
            container.appendChild(item);
        });
    }
    
    startMetricsUpdates() {
        // Load initial metrics
        this.loadMetrics();
        
        // Set up periodic updates
        this.metricsUpdateInterval = setInterval(() => {
            this.loadMetrics();
        }, 10000); // Update every 10 seconds
    }
    
    stopMetricsUpdates() {
        if (this.metricsUpdateInterval) {
            clearInterval(this.metricsUpdateInterval);
            this.metricsUpdateInterval = null;
        }
    }
    
    toggleVisualizationOption(option) {
        const checkbox = option.querySelector('.viz-checkbox');
        const isChecked = checkbox.classList.contains('checked');
        
        checkbox.classList.toggle('checked', !isChecked);
        
        // Get option type from data attribute
        const optionType = option.dataset.option;
        
        // Send update to server
        this.updateVisualizationConfig(optionType, !isChecked);
    }
    
    async updateVisualizationConfig(option, enabled) {
        try {
            const config = {};
            config[option] = enabled;
            
            await this.apiCall('/config/visualization', 'POST', config);
            
        } catch (error) {
            console.warn('Error updating visualization config:', error);
        }
    }
    
    initializeVisualizationOptions() {
        // Set default states for visualization options
        const defaultOptions = {
            'show_attention': true,
            'show_decision_path': true,
            'show_safety_zones': true,
            'show_q_values': true
        };
        
        this.elements.vizOptions.forEach(option => {
            const optionType = option.dataset.option;
            const checkbox = option.querySelector('.viz-checkbox');
            
            if (defaultOptions[optionType]) {
                checkbox.classList.add('checked');
            }
        });
    }
    
    switchMetricsTab(tabType) {
        // Update active tab
        this.elements.metricsTabs.forEach(tab => {
            tab.classList.toggle('active', tab.dataset.tab === tabType);
        });
        
        // Update content
        this.updateMetricsContent(tabType);
    }
    
    updateMetricsContent(tabType) {
        const contentContainer = document.getElementById('metricsContent');
        if (!contentContainer) return;
        
        contentContainer.innerHTML = '';
        
        switch (tabType) {
            case 'training':
                contentContainer.innerHTML = `
                    <img id="trainingMetricsImage" src="" alt="Training Metrics" style="width: 100%; height: auto;" />
                `;
                break;
                
            case 'curriculum':
                contentContainer.innerHTML = `
                    <img id="curriculumProgressImage" src="" alt="Curriculum Progress" style="width: 100%; height: auto;" />
                `;
                break;
                
            case 'comparison':
                contentContainer.innerHTML = `
                    <div id="agentComparison" class="agent-comparison"></div>
                `;
                break;
                
            default:
                contentContainer.innerHTML = `
                    <div class="metrics-placeholder">
                        <p>Select a metrics tab to view data</p>
                    </div>
                `;
        }
        
        // Reload metrics to populate new content
        this.loadMetrics();
    }
    
    handleKeyPress(event) {
        // Handle keyboard shortcuts
        if (event.ctrlKey || event.metaKey) {
            switch (event.key) {
                case 's':
                    event.preventDefault();
                    if (this.gameRunning) {
                        this.stepGame();
                    }
                    break;
                    
                case 'p':
                    event.preventDefault();
                    this.toggleAutoPlay();
                    break;
                    
                case 'r':
                    event.preventDefault();
                    if (this.gameRunning) {
                        this.stopGame();
                    } else {
                        this.startGame();
                    }
                    break;
            }
        }
        
        // Handle spacebar for step
        if (event.code === 'Space' && this.gameRunning && !this.autoPlay) {
            event.preventDefault();
            this.stepGame();
        }
    }
    
    async loadInitialData() {
        // Load agent comparison
        try {
            const comparison = await this.apiCall('/agents/compare');
            if (!comparison.error) {
                this.updateAgentComparison(comparison.comparison);
            }
        } catch (error) {
            console.warn('Error loading agent comparison:', error);
        }
        
        // Load behavior analysis
        try {
            const analysis = await this.apiCall('/analysis/behavior');
            if (!analysis.error) {
                this.updateBehaviorAnalysis(analysis);
            }
        } catch (error) {
            console.warn('Error loading behavior analysis:', error);
        }
    }
    
    updateBehaviorAnalysis(analysis) {
        // Update behavior profile display
        const container = document.getElementById('behaviorAnalysis');
        if (!container) return;
        
        const profile = analysis.behavior_profile || {};
        const insights = analysis.insights || [];
        const recommendations = analysis.recommendations || [];
        
        container.innerHTML = `
            <div class="behavior-section">
                <h4>Behavior Profile</h4>
                <p><strong>Strategy:</strong> ${profile.dominant_strategy || 'Unknown'}</p>
                <p><strong>Risk Level:</strong> ${profile.preferred_risk_level || 'Unknown'}</p>
                <p><strong>Decision Quality:</strong> ${(profile.average_decision_quality * 100).toFixed(1)}%</p>
            </div>
            
            <div class="behavior-section">
                <h4>Insights</h4>
                ${insights.map(insight => `<p>• ${insight}</p>`).join('')}
            </div>
            
            <div class="behavior-section">
                <h4>Recommendations</h4>
                ${recommendations.map(rec => `<p>• ${rec}</p>`).join('')}
            </div>
        `;
    }
    
    // Utility methods
    async apiCall(endpoint, method = 'GET', data = null) {
        const url = this.apiBase + endpoint;
        const options = {
            method,
            headers: {
                'Content-Type': 'application/json',
            },
        };
        
        if (data && method !== 'GET') {
            options.body = JSON.stringify(data);
        }
        
        try {
            const response = await fetch(url, options);
            return await response.json();
        } catch (error) {
            console.error('API call failed:', error);
            throw error;
        }
    }
    
    showError(message) {
        this.showMessage(message, 'error');
    }
    
    showSuccess(message) {
        this.showMessage(message, 'success');
    }
    
    showInfo(message) {
        this.showMessage(message, 'info');
    }
    
    showMessage(message, type = 'info') {
        // Create message element
        const messageEl = document.createElement('div');
        messageEl.className = `message ${type}-message fade-in`;
        messageEl.textContent = message;
        
        // Add to container
        let container = this.elements.errorContainer;
        if (!container) {
            container = document.body;
        }
        
        container.appendChild(messageEl);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (messageEl.parentNode) {
                messageEl.parentNode.removeChild(messageEl);
            }
        }, 5000);
    }
    
    showLoading(message = 'Loading...') {
        // Show loading indicator
        const loader = document.createElement('div');
        loader.id = 'globalLoader';
        loader.className = 'loading-overlay';
        loader.innerHTML = `
            <div class="loading-spinner"></div>
            <div class="loading-text">${message}</div>
        `;
        
        document.body.appendChild(loader);
    }
    
    hideLoading() {
        const loader = document.getElementById('globalLoader');
        if (loader) {
            loader.remove();
        }
    }
    
    cleanup() {
        // Clean up intervals and event listeners
        this.stopStatusUpdates();
        this.stopMetricsUpdates();
        this.stopAutoPlay();
    }
}

// Initialize the interface when the page loads
window.snakeAI = new SnakeAIInterface();

// Clean up on page unload
window.addEventListener('beforeunload', () => {
    if (window.snakeAI) {
        window.snakeAI.cleanup();
    }
});
