/*
 * Play with this code and it'll update in the panel opposite.
 *
 * Why not try some of the options above?
 */


Morris.Bar({
    element: 'bar-example',
    data: [
      { y: 'China', a: 116500.94},
      { y: 'India', a: 77005.77},
      { y: 'EEUU', a: 30988.08},
      { y: 'Paquistan', a: 19369.82},
      { y: 'Brasil', a: 16876.65},
      { y: 'Mexico', a: 12831.78},
      { y: 'Indones..', a: 10689.88},
      { y: 'Alemania', a: 9543.6},
      { y: 'Egipto', a: 8862.16},
      { y: 'Bangladesh', a: 8377.55}
    ],
    xkey: 'y',
    ykeys: 'a',
    labels: ['Pacientes'],
  });