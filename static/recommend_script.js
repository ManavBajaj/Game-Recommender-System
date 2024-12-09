// particles.js loading
particlesJS.load('particles-js', '/static/particles-config.js', function () {
    console.log('Particles.js loaded!');
});

// Adjust #particles-js height dynamically based on document height
function adjustParticlesHeight() {
    const particlesElement = document.getElementById('particles-js');
    if (particlesElement) {
        particlesElement.style.height = `${document.body.scrollHeight}px`;
    }
}

// Hide the clear output button initially
document.getElementById('clear-button-container').style.display = 'none';

// Show the clear output button after recommendations are displayed
if (document.getElementById('recommendations-table')) {
    document.getElementById('clear-button-container').style.display = 'block';
}

// Clear the form fields when the "Clear Form" button is clicked
document.getElementById('clear-button').addEventListener('click', function () {
    // Reset the form
    document.getElementById('game-form').reset();

    // Hide the recommendations table and header
    const table = document.getElementById('recommendations-table');
    if (table) {
        table.style.display = 'none';
    }

    const header = document.getElementById('recommendation-header');
    if (header) {
        header.style.display = 'none';
    }

    // Hide the "Clear Output" button again
    document.getElementById('clear-button-container').style.display = 'none';
});

// When the "Clear Form" button is clicked, redirect to the recommendation page (reset the form)
document.getElementById('clear-button').addEventListener('click', function () {
    // Redirect to the recommendation page to reset the form and clear recommendations
    window.location.href = "http://127.0.0.1:5000/recommendation";
});

// Clear the recommendations and header when the "Clear Output" button is clicked
document.getElementById('clear-output-button').addEventListener('click', function () {
    // Clear the recommendations table
    const table = document.getElementById('recommendations-table');
    if (table) {
        table.style.display = 'none';
    }

    // Clear the header
    const header = document.getElementById('recommendation-header');
    if (header) {
        header.style.display = 'none';
    }

    // Hide the "Clear Output" button
    document.getElementById('clear-button-container').style.display = 'none';

    // Reset the form for new input
    document.getElementById('game-form').reset();
});