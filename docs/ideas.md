Yes. With WebGL / Three.js plus the Web Audio API, you can build a highly audio-reactive music visualizer. The key parts are:

- capture audio input or an audio file
- use an analyser node to get frequency/time data
- map that data to visual parameters
- render patterns with shaders, particles, shapes, or geometry

That tech can produce many creative visualizer patterns if you design the mapping and animation carefully.