var data_per = [];
var data_loc = [];
var data_org = [];

$(document).ready(function(){
    var limitNum = 200;
    var pattern = '已输入 0/200 字符';
    $('#contentwordage').html(pattern);
    $('#input').keyup(
        function(event){
            if(event.keyCode=='32' || event.keyCode=='13'){
            var remain = $(this).val().length;
            if(remain > 200){
                pattern = '字数超过限制！';
            }else{
                data_per=[];
                data_loc=[];
                data_org=[];
                var result = remain;
                predict();
                pattern = '已输入' + result + '/200 字符';
            }
            $('#contentwordage').html(pattern);
        }}
    );
});

function predict(){
    sentence = document.getElementById('input').value;

    $.ajax({
                url: '/predict',
                type: 'post',
                contentType: 'application/json',
                data: JSON.stringify(sentence),
                success: (data) => {
                    eval('var item=' + data)
                    for(var i=0; i<item.PER.length; i++){
                        data_per[i] = item.PER[i];
                    }
                    for(var i=0; i<item.LOC.length; i++){
                        data_loc[i] = item.LOC[i];
                    }
                    for(var i=0; i<item.ORG.length; i++){
                        data_org[i] = item.ORG[i];
                    }
                    get_item()
                    }
    });
}

function get_item() {
    var add_div = document.getElementById('per');
    var html = '';
    var flag = 'per_';
    for (var i=0; i<data_per.length; i++){
        html += setDiv(data_per[i],flag+i)
    }
    add_div.innerHTML = html;

    add_div = document.getElementById('loc');
    html = '';
    flag = 'loc_';
    for (var i=0; i<data_loc.length; i++){
        html += setDiv(data_loc[i],flag+i)
    }
    add_div.innerHTML = html;

    add_div = document.getElementById('org');
    html = '';
    flag = 'org_';
    for (var i=0; i<data_org.length; i++){
        html += setDiv(data_org[i],flag+i)
    }
    add_div.innerHTML = html
}


function setDiv(item,id_name){
    var result = '<div class="result-tips" id="'+id_name+'" onclick="test(this)">'+ '<div class="result-tips-mes">' +  item.item+ '</div>'+
        '<div class="result-tips-mes">' +  item.tag + '</div>'+ '<div class="result-tips-des">'
        + '</div></div>';
    return result
}

function test(element){
    var result = element.id;
    var flag = result.slice(0,3);
    var num = result.slice(4);
    get_item();
    document.getElementById(result).className = "result-tips result-tips-show";
    if(flag=='per') {
        $("#position ").html(data_per[num].offset);
        $("#means ").html(data_per[num].tag);
        $("#length ").html(data_per[num].length)
    }else if(flag=='loc'){
        $("#position ").html(data_loc[num].offset);
        $("#means ").html(data_loc[num].tag);
        $("#length ").html(data_loc[num].length)
    }else{
        $("#position ").html(data_org[num].offset);
        $("#means ").html(data_org[num].tag);
        $("#length ").html(data_org[num].length)
    }
}


function clean(){
    window.location.reload(true);
}
