// 开启bootstrap的冒泡提示
$(function () {
  $('[data-toggle="tooltip"]').tooltip()
})

// 图片预览
$('#inputfile').bind('change', function() {
  let fileSize = this.files[0].size/1024/1024; // this gives in MB
  if (fileSize > 1) {
    $("#inputfile").val(null);
    alert('file is too big. images more than 1MB are not allowed')
    console.log('file is too big. images more than 1MB are not allowed')
    return
  }

  let ext = $('#inputfile').val().split('.').pop().toLowerCase();
  if ($.inArray(ext, ['jpg', 'jpeg', 'png']) == -1) {
    $("#inputfile").val(null);
    alert('only jpeg/jpg/png files are allowed!');
  }

  var imgFile = this.files[0];
  var fr = new FileReader();
  fr.onload = function () {
    var img = $('#imgShowHere').attr('src', fr.result)
  }
  fr.readAsDataURL(imgFile);
});

// 预测结果
$('#upload').click(function () {
  var targetUrl = $("#form").attr("action");
  var data = new FormData($("#form")[0])
  $.ajax({
    type: "post",
    url: targetUrl,
    cache: false,
    processData: false,
    contentType: false,
    data: data,
    dataType: "json",
    success: function (res) {
      console.log(res)
      $('#result').text(res.prediction)
      // $('#result').attr('data-original-title', 'Confidence: ' + res.confidence)
    },
    error: function (err) {
      alert('run error. view details at the console!')
      console.log(err)
    }
  })
})