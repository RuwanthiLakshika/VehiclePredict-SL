// RetailPredict - Main JavaScript File

/**
 * Initialize the application on page load
 */
document.addEventListener('DOMContentLoaded', function() {
    console.log('RetailPredict App Initialized');
    initializeEventListeners();
});

/**
 * Initialize event listeners for various elements
 */
function initializeEventListeners() {
    // Form submission is handled in specific pages
    // This is for global functionality
    
    // Smooth scroll behavior
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth' });
            }
        });
    });
}

/**
 * Show loading state on elements
 */
function showLoading(element) {
    if (typeof element === 'string') {
        element = document.getElementById(element);
    }
    if (element) {
        element.innerHTML = '<div class="loading"><i class="fas fa-spinner"></i> Loading...</div>';
    }
}

/**
 * Show error message
 */
function showErrorMessage(message, containerId = 'errorSection') {
    const container = document.getElementById(containerId);
    if (container) {
        container.innerHTML = `
            <div class="error-card">
                <h3><i class="fas fa-exclamation-circle"></i> Error</h3>
                <p>${message}</p>
                <button onclick="this.parentElement.parentElement.style.display='none'" class="btn btn-secondary">
                    Dismiss
                </button>
            </div>
        `;
        container.style.display = 'block';
    }
}

/**
 * Show success message
 */
function showSuccessMessage(message, containerId = 'successSection') {
    const container = document.getElementById(containerId);
    if (container) {
        container.innerHTML = `
            <div class="success-card">
                <h3><i class="fas fa-check-circle"></i> Success</h3>
                <p>${message}</p>
            </div>
        `;
        container.style.display = 'block';
    }
}

/**
 * Format currency values
 */
function formatCurrency(value) {
    return '₨ ' + value.toLocaleString('en-US', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    });
}

/**
 * Format numbers with thousands separator
 */
function formatNumber(value) {
    return value.toLocaleString('en-US', {
        minimumFractionDigits: 0,
        maximumFractionDigits: 2
    });
}

/**
 * Validate form inputs
 */
function validateForm(form) {
    const inputs = form.querySelectorAll('input[required], select[required]');
    let isValid = true;

    inputs.forEach(input => {
        if (!input.value.trim()) {
            input.classList.add('error');
            isValid = false;
        } else {
            input.classList.remove('error');
        }
    });

    return isValid;
}

/**
 * Reset form styling
 */
function resetFormStyling(form) {
    form.querySelectorAll('input, select').forEach(input => {
        input.classList.remove('error');
    });
}

/**
 * API Call utility
 */
async function apiCall(url, options = {}) {
    try {
        const response = await fetch(url, {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        console.error('API Error:', error);
        throw error;
    }
}

/**
 * Parse and display API errors
 */
function handleApiError(error) {
    let message = 'An unexpected error occurred';

    if (error.message) {
        message = error.message;
    } else if (typeof error === 'string') {
        message = error;
    }

    return message;
}

/**
 * Create a chart using Plotly
 */
function createChart(containerId, data, layout = {}) {
    const defaultLayout = {
        responsive: true,
        hovermode: 'x unified',
        template: 'plotly_white',
        margin: { t: 30, b: 30, l: 50, r: 30 },
        ...layout
    };

    Plotly.newPlot(containerId, data, defaultLayout, { responsive: true });
}

/**
 * Create a bar chart
 */
function createBarChart(containerId, labels, values, title = '', xLabel = '', yLabel = '') {
    const trace = {
        x: labels,
        y: values,
        type: 'bar',
        marker: { color: '#0066CC' }
    };

    const layout = {
        title: title,
        xaxis: { title: xLabel },
        yaxis: { title: yLabel },
        height: 300,
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)'
    };

    createChart(containerId, [trace], layout);
}

/**
 * Create a horizontal bar chart
 */
function createHorizontalBarChart(containerId, labels, values, title = '') {
    const trace = {
        x: values,
        y: labels,
        type: 'bar',
        orientation: 'h',
        marker: { color: '#0066CC' }
    };

    const layout = {
        title: title,
        xaxis: { title: 'Value' },
        height: Math.max(300, labels.length * 30),
        margin: { l: 150 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)'
    };

    createChart(containerId, [trace], layout);
}

/**
 * Create a line chart
 */
function createLineChart(containerId, x, y, title = '', xLabel = '', yLabel = '') {
    const trace = {
        x: x,
        y: y,
        type: 'scatter',
        mode: 'lines+markers',
        line: { color: '#0066CC', width: 2 },
        marker: { size: 5 }
    };

    const layout = {
        title: title,
        xaxis: { title: xLabel },
        yaxis: { title: yLabel },
        height: 300,
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)'
    };

    createChart(containerId, [trace], layout);
}

/**
 * Scroll to element smoothly
 */
function scrollToElement(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
}

/**
 * Toggle loading spinner
 */
function toggleSpinner(show = true) {
    let spinner = document.getElementById('globalSpinner');

    if (show) {
        if (!spinner) {
            spinner = document.createElement('div');
            spinner.id = 'globalSpinner';
            spinner.style.cssText = `
                position: fixed;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                z-index: 9999;
                background: white;
                padding: 2rem;
                border-radius: 12px;
                box-shadow: 0 8px 24px rgba(0,0,0,0.15);
                text-align: center;
            `;
            spinner.innerHTML = '<i class="fas fa-spinner fa-spin" style="font-size: 2rem; color: #0066CC;"></i>';
            document.body.appendChild(spinner);
        }
        spinner.style.display = 'block';
    } else if (spinner) {
        spinner.style.display = 'none';
    }
}

/**
 * Deep copy an object
 */
function deepCopy(obj) {
    return JSON.parse(JSON.stringify(obj));
}

/**
 * Debounce function for input events
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

/**
 * Get query parameters from URL
 */
function getQueryParam(param) {
    const searchParams = new URLSearchParams(window.location.search);
    return searchParams.get(param);
}

/**
 * Set page title
 */
function setPageTitle(title) {
    document.title = title + ' - RetailPredict';
}

/**
 * Log analytics event (placeholder for future analytics)
 */
function logEvent(eventName, eventData = {}) {
    console.log(`Event: ${eventName}`, eventData);
    // Can be connected to Google Analytics or other tracking services
}
