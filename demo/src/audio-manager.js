/**
 * Convert time in seconds to 'HH:MM'
 */
const numToTime = num => {
  let minutes = String(Math.min(99, Math.floor(num / 60)));
  let seconds = String(Math.round(num % 60));
  minutes = minutes.length === 2 ? minutes : `0${minutes}`;
  seconds = seconds.length === 2 ? seconds : `0${seconds}`;
  return `${minutes}:${seconds}`;
};

export default class AudioManager {

  constructor() {
    // Audio element setup
    this.audio = new Audio();
    this.audio.preload = 'metadata';
    this.audio.crossOrigin = 'anonymous';
    this.context = new AudioContext();
    this.source = this.context.createMediaElementSource(this.audio);
    this.source.connect(this.context.destination);

    // State
    this.paused = true;
    this.active = false;
    this.trackbarDrag = false;
    this.trackbar_timestamp = null;

    // Track control elements
    this.vizBox = document.getElementById('viz-box');
    this.vizControl = document.getElementById('viz-control');
    this.playButton = document.getElementById('viz-control-play');
    this.pauseButton = document.getElementById('viz-control-pause');
    this.stopButton = document.getElementById('viz-control-stop');
    this.trackBar = document.getElementById('viz-control-trackbar');
    this.timeElapsed = document.getElementById('viz-control-time-elapsed');
    this.timeLeft = document.getElementById('viz-control-time-left');
    this.initTrackControl();

    // Callbacks
    this.onPlay = null;
    this.onPause = null;
    this.onStop = null;
  }

  loadTrack(src, onloadedmetadata) {
    // Use the callback to access track metadata, for example, duration
    this.audio.onloadedmetadata = onloadedmetadata;
    this.audio.src = src;
  }

  activate() {
    this.active = true;
    this.trackbar_timestamp = Date.now();
  }

  deactivate() {
    this.active = false;
    this.trackbar_timestamp = null;
  }

  play() {
    if (this.paused) {// Ignore otherwise
      this.audio.play();
      this.paused = false;
      // Show pause button
      this.playButton.style.display = 'none';
      this.pauseButton.style.display = 'block';
      // Callback
      this.onPlay && this.onPlay();
    }
  }

  pause() {
    if (!this.paused) {// Ignore otherwise
      this.audio.pause();
      this.paused = true;
      // Show play button
      this.pauseButton.style.display = 'none';
      this.playButton.style.display = 'block';
      // Callback
      this.onPause && this.onPause();
    }
  }

  stop() {
    if (this.active){// Ignore otherwise
      // Pause track and reset current time
      this.pause();
      this.audio.currentTime = 0;
      // Callback
      this.onStop && this.onStop();
    }
  }

  initTrackControl() {
    // Connect button actions to handlers
    this.playButton.onclick = () => this.play();
    this.pauseButton.onclick = () => this.pause();
    this.stopButton.onclick = () => this.stop();

    // Update track bar
    this.updateTrackBar();

    // Drag track bar
    this.trackBar.onmousedown = () => this.trackbarDrag = true;
    this.trackBar.onmouseup = () => {
      if (this.active && this.audio.duration) {// Drag is only allowed when a track is active
        this.audio.currentTime = this.audio.duration * this.trackBar.value; // trackbar range is [0, 1]
      }
      this.trackbarDrag = false;
    };

    // Reset user active timestamp on mouse move
    this.vizBox.onmousemove = () => this.active && (this.trackbar_timestamp = Date.now());

    // Play/Pause on
    this.vizBox.onclick = () => this.active && (this.paused ? this.play() : this.pause());
    this.vizControl.ondblclick = ev => ev.stopPropagation();
    this.vizControl.onclick = ev => ev.stopPropagation();

    // Track end
    this.audio.onended = () => this.stop();
  }

  updateTrackBar() {
    requestAnimationFrame(() => this.updateTrackBar());

    // Update Duration
    if (this.active && this.audio.duration) {
      if (!this.trackbarDrag) {
        this.trackBar.value = this.audio.currentTime / this.audio.duration;
        this.timeElapsed.querySelector('div').innerText = numToTime(this.audio.currentTime);
        this.timeLeft.querySelector('div').innerText = numToTime(this.audio.duration - this.audio.currentTime);
      }
    } else {
      this.trackBar.value = 0;
      this.timeElapsed.querySelector('div').innerText = '00:00';
      this.timeLeft.querySelector('div').innerText = '00:00';
    }

    // Display trackbar only when active
    let vizIactive = !this.active;
    if (vizIactive) {
      this.vizBox.classList.add('viz-inactive');
    } else {
      this.vizBox.classList.remove('viz-inactive');
    }

    // Fade trackbar (and hide mouse) when user inactive
    let userInactive = this.active && (this.trackbar_timestamp && Date.now() - this.trackbar_timestamp > 3000);
    if (userInactive) {
      this.vizBox.classList.add('viz-user-inactive');
    } else {
      this.vizBox.classList.remove('viz-user-inactive');
    }
  }

}
