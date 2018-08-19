const message_loading = '<span>' +
  'Extracting features from the track...' +
  '<br>' +
  '<i class="fa fa-2x fa-circle-o-notch fa-spin"></i>&nbsp;' +
  '</span>';

const message_error = 'Oops! Something went terribly wrong ' +
  '<i class="fa fa-2x fa-frown-o"></i>' +
  '<br>' +
  'Check your internet connection, perhaps?';


export default class VizBoxMessages {

  constructor() {
    this.messageContainer = document.getElementById("message-box");
    this.init_message = this.messageContainer.innerHTML;
  }

  updateMessage(show, message) {
    // Toggle visibility
    if (show) {
      this.messageContainer.style.display = '';
    } else if (!show) {
      this.messageContainer.style.display = 'none';
    }
    // Render new message
    if (message) {
      this.messageContainer.innerHTML = message;
    }
  }

  hide() {
    this.updateMessage(false);
  }

  setInit() {
    this.updateMessage(true, this.init_message);
  }

  setLoading() {
    this.updateMessage(true, message_loading);
  }

  setError() {
    this.updateMessage(true, message_error);
  }
}
