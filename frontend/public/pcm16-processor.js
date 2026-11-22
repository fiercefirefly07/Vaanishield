
class PCM16Processor extends AudioWorkletProcessor {
  constructor() {
    super();

    this.targetRate = 16000;
    this.step = sampleRate / this.targetRate;

    this.inputFrame = 0;
    this.nextOutputFrame = 0;
    this.previousSample = 0;

    // 0.5 seconds of 16 kHz audio
    this.chunkSize = 8000;
    this.chunk = new Int16Array(this.chunkSize);
    this.chunkPosition = 0;
  }

  process(inputs) {
    const channels = inputs[0];

    if (!channels || channels.length === 0) {
      return true;
    }

    const length = channels[0].length;

    for (let i = 0; i < length; i++) {
      let current = 0;

      for (const channel of channels) {
        current += channel[i];
      }

      current /= channels.length;

      if (this.inputFrame === 0) {
        this.previousSample = current;
      }

      // Resample continuously to 16 kHz
      while (this.nextOutputFrame <= this.inputFrame) {
        let value = current;

        if (this.inputFrame > 0) {
          const fraction = Math.max(
            0,
            Math.min(
              1,
              this.nextOutputFrame - (this.inputFrame - 1)
            )
          );

          value =
            this.previousSample +
            (current - this.previousSample) * fraction;
        }

        value = Math.max(-1, Math.min(1, value));

        const pcm = Math.round(
          value < 0 ? value * 32768 : value * 32767
        );

        this.chunk[this.chunkPosition++] = pcm;

        if (this.chunkPosition === this.chunkSize) {
          const buffer = this.chunk.buffer;

          this.port.postMessage(buffer, [buffer]);

          this.chunk = new Int16Array(this.chunkSize);
          this.chunkPosition = 0;
        }

        this.nextOutputFrame += this.step;
      }

      this.previousSample = current;
      this.inputFrame++;
    }

    return true;
  }
}

registerProcessor("pcm16-processor", PCM16Processor);
