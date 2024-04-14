
let mapDivId = 'map'

document.addEventListener('DOMContentLoaded', () => {
  const inputImage = document.querySelector('input#inputImage')

  inputImage.addEventListener('input', ({target:input}) => {
    const file = input.files[0]
  })
  
  const map = L.map(mapDivId, {
    crs: L.CRS.Simple,
    zoom: 0,
    center: [0,0]
  })

  L.tileLayer('/tiles/{filename}_{z}_{x}_{y}', {
    filename: '_',
    minZoom: -2,
    maxZoom: 60,
    attribution: null
  }).addTo(map)

  map.setView([0,0], 0)

  console.log(map)
})
