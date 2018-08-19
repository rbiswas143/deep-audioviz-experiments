import {api_url} from "./client";

import $ from 'jquery';
import 'selectize';
import '../node_modules/selectize/dist/css/selectize.css';

const devMode = true;

// Value for All Tracks and All Genres, chosen for alphabetical precedence
const ALL = '(ALL)';

export default class FMATrackSelector {

  constructor() {
    this.fmaTracks = null; // Init from server

    // UI Elements
    this.$fmaGenreSelect = $('#fma-genre-select');
    this.$fmaArtistSelect = $('#fma-artist-select');
    this.$fmaTrackSelect = $('#fma-track-select');
    this.fmaLoaderButton = $('#fma-loader-btn');
    this.$serverStatusButton = $('#server-status');


    // Fetch FMA metadata from backend
    this.fetchTracks(
      tracksData => this.onMetadataLoadSuccess(tracksData),
      error => this.onMetadataLoadFail(error)
    );

  }

  fetchTracks(onSuccess, onFailure) {
    fetch(`${api_url}/fetchtracks`, {
      method: "GET"
    })
      .then(response => response.json())
      .then(onSuccess, onFailure)
  }

  onMetadataLoadSuccess(tracksData) {
    // Set server state
    this.$serverStatusButton.find('i').addClass('green');
    this.$serverStatusButton.attr('title', 'Sever is UP!   \\[]/');

    // Convert metadata to array of objects as needed by selectize
    this.fmaTracks = [];
    tracksData.forEach(data => {
      this.fmaTracks.push({
        id: data[0],
        title: data[1],
        artist: data[2],
        genre: data[3],
      });
    });
    // Init Modal
    this.initModal();
    // Dev Mode
    if (devMode) {
      document.getElementById('fma-loader-btn').click();
      document.getElementById('go-btn').click();
    }
  }

  initModal() {

    // Init Genre (default: ALL)
    this.$fmaGenreSelect.selectize({
      options: this.fmaTracks,
      labelField: 'genre',
      valueField: 'genre',
      sortField: 'genre',
      searchField: ['genre']
    });
    this.$fmaGenreSelect[0].selectize.addOption({genre: ALL});
    this.$fmaGenreSelect[0].selectize.setValue(ALL);

    // Init Artist (default: ALL)
    this.$fmaArtistSelect.selectize({
      labelField: 'artist',
      valueField: 'artist',
      sortField: 'artist',
      searchField: ['artist'],
    });
    this.updateArtists(this.fmaTracks);

    // Init Track (default: first option)
    this.$fmaTrackSelect.selectize({
      labelField: 'title',
      valueField: 'id', // Track ID is used to
      sortField: 'title',
      searchField: ['title'],
    });
    this.updateTracks(this.fmaTracks);

    // Genre change handler (filter artists and tracks)
    this.$fmaGenreSelect.change(() => {
      const genre = this.$fmaGenreSelect[0].selectize.getValue();
      if (genre === ALL || !genre) { // Empty treated as ALL
        // Remove filters
        this.updateArtists(this.fmaTracks);
        this.updateTracks(this.fmaTracks)
      } else {
        // Filter genre
        const filteredTracks = this.fmaTracks.filter(track => track.genre === genre);
        this.updateArtists(filteredTracks);
        this.updateTracks(filteredTracks)
      }
    });

    // Artist change handler (filter tracks)
    this.$fmaArtistSelect.change(() => {
      const artist = this.$fmaArtistSelect[0].selectize.getValue();
      if (artist === ALL || !artist) { // Empty treated as ALL
        // Delegate to genre change handler to filter ALL tracks of currently selected genre
        this.$fmaGenreSelect.change();
      } else {
        // Filter artist
        this.updateTracks(this.fmaTracks.filter(track => track.artist === artist));
      }
    });

  }

  updateTracks(filteredTracks) {
    this.$fmaTrackSelect[0].selectize.setValue(null, true);
    this.$fmaTrackSelect[0].selectize.clearOptions();
    this.$fmaTrackSelect[0].selectize.addOption(filteredTracks);
    this.$fmaTrackSelect[0].selectize.setValue(filteredTracks[0].id, true); // Default: first option
  }

  updateArtists(filteredTracks) {
    this.$fmaArtistSelect[0].selectize.setValue(null, true);
    this.$fmaArtistSelect[0].selectize.clearOptions();
    this.$fmaArtistSelect[0].selectize.addOption({artist: ALL});
    this.$fmaArtistSelect[0].selectize.addOption(filteredTracks);
    this.$fmaArtistSelect[0].selectize.setValue(ALL, true); //Default: ALL
  }

  onMetadataLoadFail(error) {
    // Disable FMA loader modal
    console.log("Error:", error);
    let els = document.querySelectorAll('.fma-metadata-fail');
    for (let i = 0; i < els.length; i++) {
      els[i].style.display = '';
    }
    els = document.querySelectorAll('.fma-metadata-success');
    for (let i = 0; i < els.length; i++) {
      els[i].style.display = 'none';
    }
  }

  getCurrFMATrack() {
    const track_id = this.$fmaTrackSelect[0].selectize.getValue();
    if (!track_id) return;
    const currTrackData = this.fmaTracks.filter(track => track.id == track_id)[0];
    return {
      track_id: track_id,
      title: currTrackData.title,
      artist: currTrackData.artist,
      genre: currTrackData.genre,
      src: `${api_url}/downloadtrack/${track_id}`,
      type: 'fma'
    };
  }

  getTrackDisplayName(track) {
    return `[FMA] ${track.title} by ${track.artist} [${track.genre}]`;
  }
}
