$('.form-files').on('submit', function () {
  let self = $(this),
      btnSubmit = self.find('input[type="submit"], button');
    btnSubmit.attr('disabled', 'disabled');
});