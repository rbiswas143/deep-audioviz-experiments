export default class AudioManager {

  constructor() {
    this.audio = new Audio();
    this.audio.preload = 'metadata';
    this.audio.crossOrigin = 'anonymous';

    this.context = new AudioContext();
    this.source = this.context.createMediaElementSource(this.audio);
    this.source.connect(this.context.destination);
    this.paused = true;
  }

  loadTrack(src, onloadedmetadata) {
    this.audio.onloadedmetadata = onloadedmetadata;
    this.audio.src = src;
  }

  play() {
    this.audio.play();
    this.paused = false;
  }

  pause() {
    this.audio.pause();
    this.paused = true;
  }
}
