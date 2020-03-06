function predict(){
    sentence = document.getElementById('input').value;

    $.ajax({
                url: '/predict',
                type: 'post',
                contentType: 'application/json',
                data: JSON.stringify(sentence),
                success: (data) => {
                    $('#output').text(data);
                    }
                    });

}

function clean(){
    alert('delete')
    window.location.reload(true)
}