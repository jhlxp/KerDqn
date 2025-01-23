#include <linux/mm.h>
#include <linux/module.h>
#include <linux/math64.h>
#include <net/tcp.h>
#include <linux/init.h>
#include <linux/types.h>
#include <net/sock.h>
#include <linux/netlink.h>

#define kerdqn_NETLINK_TEST     23
#define kerdqn_USER_PORT    8023

#define TRAINING_TIME    80
static const u32 kerdqn_min_rtt_win_sec __read_mostly = 30;
static const u32 kerdqn_max_thr_win_sec __read_mostly = 30;

extern struct net init_net;

static char kerdqn_netlink_kmsg[64];
static struct sock *kerdqn_nlsk = NULL;
static u32 kerdqn_action = 8;
static u32 kerdqn_cwnd = 10;

enum kerdqn_mode {
	KERDQN_START,	
	KERDQN_TRAINING,		
	KERDQN_PROBE_RTT,	
	KERDQN_REDUCT
};

struct kerdqn_bictcp {
	u32 start_flag:1,
        recovery_flag:1,
        rtt_probe:1,
	init_cwnd_flag:1,
        mode:3,
        times:3,
	start_times:3,
        recovery_count:3,
	keep_times:8,
        unused:8;
	u32 min_rtt_us; 
	u32 min_rtt_stamp;
	u32 max_bw_bps; 
	u32 max_bw_stamp;
	u32 last_update_stamp;
	u32 last_slow_start_stamp;
	u32 train_count;
    	u32 bw;
	u32 bw_last;
    	u32 rtt;
	u32 rtt_last;
	u32 probe_rtt_list[3];
	u32 history_rtt[3];
	u32 times_stamp;
	u32 before_probe_cwnd;
	u32 loss_stamp;
};

static void kerdqn_tcp_init(struct sock *sk)
{
	struct kerdqn_bictcp *qc = inet_csk_ca(sk);
	struct tcp_sock *tp = tcp_sk(sk);

	tp -> snd_cwnd = 10;

	tp -> snd_ssthresh = TCP_INFINITE_SSTHRESH;
	qc -> mode = KERDQN_START;
	qc -> init_cwnd_flag = 0;
	qc -> rtt_probe = 0;
	qc -> min_rtt_us = 0xffffffff;
	qc -> min_rtt_stamp = tcp_jiffies32;
	qc -> max_bw_bps = 0;
	qc -> max_bw_stamp = tcp_jiffies32;
	qc -> last_update_stamp = tcp_jiffies32;
	qc -> last_slow_start_stamp = tcp_jiffies32;
	qc -> train_count = 0;
	qc -> before_probe_cwnd = 10;
    	qc -> bw = 0;
    	qc -> rtt = 0;
    	qc -> bw_last = 0;
    	qc -> rtt_last = 0;
	qc -> recovery_count = 0;
	qc -> recovery_flag = 0;
	qc -> probe_rtt_list[0] = 0xffffffff;
	qc -> probe_rtt_list[1] = 0xffffffff;
	qc -> probe_rtt_list[2] = 0xffffffff;
	qc -> history_rtt[0] = 0xffffffff;
	qc -> history_rtt[1] = 0xffffffff;
	qc -> history_rtt[2] = 0xffffffff;
	qc -> start_flag = 0;
	qc -> times = 0;
	qc -> start_times = 0;
	qc -> times_stamp = tcp_jiffies32;
	qc -> loss_stamp = tcp_jiffies32;
	qc -> keep_times = 0;
}

static u32 kerdqn_tcp_recalc_ssthresh(struct sock *sk)
{
	return TCP_INFINITE_SSTHRESH; 
}

static void kerdqn_get_minrtt_maxbw(struct sock *sk, const struct rate_sample *rs){

	struct kerdqn_bictcp *qc = inet_csk_ca(sk);
	bool filter_expired_minrtt;
	bool filter_expired_maxthr;
	u64 bw;
	
	if( rs -> rtt_us < 0 ){
		return ;
	}
	
	filter_expired_minrtt = after(tcp_jiffies32, qc -> min_rtt_stamp + kerdqn_min_rtt_win_sec * HZ);
	filter_expired_maxthr = after(tcp_jiffies32, qc -> max_bw_stamp + kerdqn_max_thr_win_sec * HZ);
	
	if( rs->interval_us < 0 ){
		return ;
	}
	
	bw = div64_long((u64)rs->delivered * (1 << 24), rs->interval_us);
	
	qc -> bw = (u32)bw;
	qc -> rtt = (u32)rs -> rtt_us;
	
	if (qc -> start_flag == 0 && qc -> rtt > 0){
		qc -> history_rtt[0] = qc -> rtt;
		qc -> history_rtt[1] = qc -> rtt;
		qc -> history_rtt[2] = qc -> rtt;
		qc -> start_flag = 1;
	}
	
	qc -> history_rtt[0] = qc -> history_rtt[1];
	qc -> history_rtt[1] = qc -> history_rtt[2];
	qc -> history_rtt[2] = qc -> rtt;
	
	if (rs -> rtt_us > 0 &&
	    (rs -> rtt_us < qc -> min_rtt_us ||
	     (filter_expired_minrtt && !rs->is_ack_delayed))) {
		qc -> min_rtt_us = rs->rtt_us;		
		qc -> min_rtt_stamp = tcp_jiffies32;
	}
	
	if (rs -> rtt_us > 0 &&
	    (qc -> max_bw_bps < bw ||
	     (filter_expired_maxthr && !rs->is_ack_delayed))) {
		qc -> max_bw_bps = bw;
		qc -> max_bw_stamp = tcp_jiffies32;
	}
}

static int kerdqn_send_usrmsg(char *pbuf, uint16_t len)
{
	struct sk_buff *nl_skb;
	struct nlmsghdr *nlh;  
	
	int ret;
	
	nl_skb = nlmsg_new(len, GFP_ATOMIC);
	if(!nl_skb)
	{
	printk("netlink alloc failure\n");
	return -1;
	}
	
	nlh = nlmsg_put(nl_skb, 0, 0, kerdqn_NETLINK_TEST, len, 0);
	
	
	if(nlh == NULL)
	{
	printk("nlmsg_put failaure \n");
	nlmsg_free(nl_skb);  
	return -1;
	}
	
	memcpy(nlmsg_data(nlh), pbuf, len);
	ret = netlink_unicast(kerdqn_nlsk, nl_skb, kerdqn_USER_PORT, MSG_DONTWAIT);
	
	return ret;
}

static void kerdqn_send_bw_rtt(u32 mode, u32 utility, u32 bw,u32 rtt, u32 max_bw, u32 min_rtt, u32 cwnd, u32 inflight, u32 losses, u32 loss_ratio){
	char *kmsg;
	snprintf(kerdqn_netlink_kmsg, sizeof(kerdqn_netlink_kmsg), "(%u;%u;%u;%u;%u;%u;%u;%u;%u;%u;%u;)", mode,utility,bw,rtt,max_bw,min_rtt,cwnd,inflight,losses,loss_ratio);
	kmsg = kerdqn_netlink_kmsg;
	kerdqn_send_usrmsg(kmsg, strlen(kmsg));
}

static void kerdqn_netlink_rcv_msg(struct sk_buff *skb)
{
	struct nlmsghdr *nlh = NULL;
	char *umsg = NULL;
	u32 action_receive = 0;
	
	if(skb->len >= nlmsg_total_size(0))
	{
	
	nlh = nlmsg_hdr(skb);  
	
		// printk("nlh->nlmsg_pid:%u,nlmsg_len(nlh):%u",nlh->nlmsg_pid, nlmsg_len(nlh));
	
		if (nlmsg_len(nlh) == sizeof(u32))  
		{
			u32 *data_ptr = (u32 *)NLMSG_DATA(nlh);  
			action_receive = *data_ptr;              
			printk("Received action: %u", action_receive);
	
			kerdqn_action = action_receive;
	
			switch(kerdqn_action)
			{
			case 0 :
				kerdqn_cwnd = max(10, kerdqn_cwnd - 8 <= 10 ? kerdqn_cwnd + 8 : kerdqn_cwnd - 8);
				break;
			case 1 :
				kerdqn_cwnd = max(10, kerdqn_cwnd - 7 <= 10 ? kerdqn_cwnd + 7 : kerdqn_cwnd - 7);
				break;
			case 2 :
				kerdqn_cwnd = max(10, kerdqn_cwnd - 6 <= 10 ? kerdqn_cwnd + 6 : kerdqn_cwnd - 6);
				break;
			case 3 :
				kerdqn_cwnd = max(10, kerdqn_cwnd - 5 <= 10 ? kerdqn_cwnd + 5 : kerdqn_cwnd - 5);
				break;
			case 4 :
				kerdqn_cwnd = max(10, kerdqn_cwnd - 4 <= 10 ? kerdqn_cwnd + 4 : kerdqn_cwnd - 4);
				break;
			case 5 :
				kerdqn_cwnd = max(10, kerdqn_cwnd - 3 <= 10 ? kerdqn_cwnd + 3 : kerdqn_cwnd - 3);
				break;			
			case 6 :
				kerdqn_cwnd = max(10, kerdqn_cwnd - 2 <= 10 ? kerdqn_cwnd + 2 : kerdqn_cwnd - 2);
				break;
			case 7 :
				kerdqn_cwnd = max(10, kerdqn_cwnd - 1 <= 10 ? kerdqn_cwnd + 1 : kerdqn_cwnd - 1);
				break;
			case 8 :
				kerdqn_cwnd = max(10, kerdqn_cwnd );
				break;			
			case 9 :
				kerdqn_cwnd = max(10, kerdqn_cwnd + 1);
				break;
			case 10 :
				kerdqn_cwnd = max(10, kerdqn_cwnd + 2);
				break;
			case 11 :
				kerdqn_cwnd = max(10, kerdqn_cwnd + 3);
				break;
			case 12 :
				kerdqn_cwnd = max(10, kerdqn_cwnd + 4);
				break;
			case 13 :
				kerdqn_cwnd = max(10, kerdqn_cwnd + 5);
				break;
			case 14 :
				kerdqn_cwnd = max(10, kerdqn_cwnd + 6);
				break;
			case 15 :
				kerdqn_cwnd = max(10, kerdqn_cwnd + 7);
				break;
			case 16 :
				kerdqn_cwnd = max(10, kerdqn_cwnd + 8);
				break;
			default:
				kerdqn_cwnd = max(10, kerdqn_cwnd );
			}
	
		}
	}
}


static void kerdqn_tcp_training(struct sock *sk, const struct rate_sample *rs)
{
	struct tcp_sock *tp = tcp_sk(sk);
	struct kerdqn_bictcp *qc = inet_csk_ca(sk);

	u32 count = 0;
	u32 min_value = 0;
	u32 min_cwnd = 10;
	int i = 0;
	bool times_expired=0;
	u32 loss_ratio = 0;
	bool loss_timer_expired = 0;

	if(qc -> rtt_probe == 1){
		qc -> train_count--;
		qc -> probe_rtt_list[qc->train_count] = rs -> rtt_us;
		if(qc -> train_count == 0){
			qc -> rtt_probe = 0;
			tp -> snd_cwnd = qc -> before_probe_cwnd;
			qc -> recovery_count = 3;
			qc -> recovery_flag = 1;

			min_value = qc -> probe_rtt_list[0];
			for(i=0;i<count;i++){
				if(min_value>=qc->probe_rtt_list[i]){
					min_value = qc->probe_rtt_list[i];
				}
			}
			if(min_value > 0){
				qc -> min_rtt_us = min_value;
			}
			for(i=0;i<count;i++){
					qc -> probe_rtt_list[i] = 0xffffffff;
			}
		}
	}

	if(qc -> recovery_flag == 1){

		qc -> recovery_count--;
		if(qc -> recovery_count == 0){
			qc -> recovery_flag = 0;
		}
	}

	if(qc -> rtt_probe == 0){
		if(qc -> train_count > 245 ){
			qc -> rtt_probe = 1;
			qc -> min_rtt_stamp = tcp_jiffies32;
			qc -> train_count = 2;
			count = qc -> train_count;
			qc -> before_probe_cwnd = tp -> snd_cwnd;
			tp -> snd_cwnd = max(tp -> snd_cwnd / 20, min_cwnd);
		}
		else{
			qc -> train_count++;
		}
	}

	// Time delay surge, drop the cwnd
	if ((qc->mode==1)
			&&(((qc -> history_rtt[0]-qc -> min_rtt_us) > ((qc -> min_rtt_us * 204 * 3) >> 10))) 
					&& (((qc -> history_rtt[1]-qc -> min_rtt_us) > ((qc -> min_rtt_us * 204 * 3) >> 10))) 
							&& (((qc -> history_rtt[2]-qc -> min_rtt_us) > ((qc -> min_rtt_us * 204 * 3) >> 10)))){

		times_expired = after(tcp_jiffies32, qc -> times_stamp + msecs_to_jiffies(500)); 

		if(times_expired){
			qc -> times = 0;
			qc -> times_stamp = tcp_jiffies32;
		}

		qc -> times = qc -> times + 1;

		if(qc -> times > 5){
			qc -> mode = KERDQN_REDUCT;
			qc -> times = 0;
			qc -> times_stamp = tcp_jiffies32;

			if(qc -> rtt_probe == 0 & qc -> train_count < 230){
				qc -> train_count = 230;
			}
		}

	}

	if(qc -> mode == KERDQN_REDUCT){
		tp -> snd_cwnd = (717 * tp -> snd_cwnd) >> 10;
		kerdqn_cwnd = tp -> snd_cwnd;
		qc -> mode = KERDQN_TRAINING;
		qc -> start_flag = 0;
		return ;
	}


	if(qc -> rtt_probe == 0 && qc -> recovery_flag == 0){
		if(rs -> delivered + rs -> losses <=0 ){
			loss_ratio = 0;
		}
		else{
			loss_ratio = rs -> losses * 10000 / ( rs -> delivered + rs -> losses);
		}

		loss_timer_expired = after(tcp_jiffies32, qc -> loss_stamp + msecs_to_jiffies(1000));

		if (rs -> losses > 3 && loss_timer_expired==1){
			// 3% loss_ratio
			if(loss_ratio >= 315){
				tp -> snd_cwnd = (922 * tp -> snd_cwnd) >> 10;
				kerdqn_cwnd = tp -> snd_cwnd;
				qc -> loss_stamp = tcp_jiffies32;
			}
			else if(loss_ratio >= 210){
				tp -> snd_cwnd = (922 * tp -> snd_cwnd) >> 10;
				kerdqn_cwnd = tp -> snd_cwnd;
				qc -> loss_stamp = tcp_jiffies32;
			}
			else if (loss_ratio >= 105){
				tp -> snd_cwnd = (922 * tp -> snd_cwnd) >> 10;
				kerdqn_cwnd = tp -> snd_cwnd;
				qc -> loss_stamp = tcp_jiffies32;
			}
		}

		kerdqn_send_bw_rtt((u32)qc->mode,0,qc->bw, qc->rtt,qc->max_bw_bps, qc->min_rtt_us, tp->snd_cwnd, rs->prior_in_flight, rs -> losses, loss_ratio);
	}

	qc -> last_update_stamp = tcp_jiffies32;


	if(qc -> rtt_probe == 0){
        tp -> snd_cwnd = kerdqn_cwnd;
    }

}

static void kerdqn_slow_start(struct sock *sk, const struct rate_sample *rs)
{
	struct tcp_sock *tp = tcp_sk(sk);
	struct kerdqn_bictcp *qc = inet_csk_ca(sk);
	u64 ratio_bw = 100;
	u64 ratio_rtt = 100;
	u64 utility = 100;
	u32 cwnd = kerdqn_cwnd;
	u32 loss_ratio = 0;

	if (qc -> bw <= 0)
		return ;
	else if (qc -> bw_last <= 0){
		qc -> bw_last = qc -> bw;
	}

	if (qc -> rtt <= 0)
		return ;
	else if (qc -> rtt_last <= 0){
		qc -> rtt_last = qc -> rtt;	
	}

	ratio_bw = qc -> bw * 100 / qc -> bw_last;
	ratio_rtt = qc -> rtt * 100 / qc -> rtt_last;

	utility = ratio_bw * 100 / ratio_rtt;

	// printk("(utility:%lld)(bw:%u)(bw_last:%u)(rtt:%u)(rtt_last:%u)",utility,qc -> bw,qc -> bw_last,qc -> rtt,qc -> rtt_last);

	if (utility >= 99){

		if (qc -> rtt < (qc -> min_rtt_us * 11 / 10)){
			cwnd = cwnd + max(2, cwnd / 10);
		} 

		if (qc -> start_times > 0){
			qc -> start_times = qc -> start_times - 1;
		}

	}
	else{

		qc -> start_times = qc -> start_times + 1;

		if (qc -> rtt > (qc -> min_rtt_us * 11 / 10)){
			cwnd = cwnd - max(2, cwnd / 15);
		} 

		if (qc -> start_times >= 3){

			qc -> mode = KERDQN_TRAINING;

		}
	}

	cwnd = max(cwnd, 10);
	cwnd = min(cwnd, tp -> snd_cwnd_clamp);

	tp -> snd_cwnd = cwnd;
	kerdqn_cwnd = tp -> snd_cwnd;

	if(rs -> delivered + rs -> losses <=0 ){
		loss_ratio = 0;
	}
	else{
		loss_ratio = rs -> losses * 10000 / ( rs -> delivered + rs -> losses);
	}

	// loss higher
	if(loss_ratio >= 1000){
		tp -> snd_cwnd = (922 * tp -> snd_cwnd) >> 10;
		kerdqn_cwnd = tp -> snd_cwnd;
		qc -> loss_stamp = tcp_jiffies32;
	}

	kerdqn_send_bw_rtt((u32)qc->mode,(u32)utility,qc->bw, qc->rtt,qc->max_bw_bps, qc->min_rtt_us, tp->snd_cwnd, rs->prior_in_flight, rs -> losses, loss_ratio);
	
	qc -> last_slow_start_stamp = tcp_jiffies32;

	qc -> bw_last  = qc -> bw;
	qc -> rtt_last = qc -> rtt;
}

static void kerdqn_tcp_cong_main(struct sock *sk, const struct rate_sample *rs)
{
	struct tcp_sock *tp = tcp_sk(sk);
	struct kerdqn_bictcp *qc = inet_csk_ca(sk);
	bool training_timer_expired = after(tcp_jiffies32, qc -> last_update_stamp + msecs_to_jiffies(TRAINING_TIME)); 
	bool slowstart_timer_expired = after(tcp_jiffies32, qc -> last_slow_start_stamp + msecs_to_jiffies(20));

	if (qc -> init_cwnd_flag == 0 && qc->mode == KERDQN_START){
		kerdqn_cwnd = 10;
		qc -> init_cwnd_flag = 1;
	}

	// Performance monitoring
	kerdqn_get_minrtt_maxbw(sk,rs);

	if(qc -> rtt_probe == 0){
        tp -> snd_cwnd = kerdqn_cwnd;
    }

	if (slowstart_timer_expired && qc->mode == KERDQN_START){
		kerdqn_slow_start(sk, rs);
	}
	else if (training_timer_expired && qc -> mode == KERDQN_TRAINING){
		kerdqn_tcp_training(sk, rs);
	}

	// if(qc -> bw + 5000 * 2 > qc -> max_bw && qc -> rtt < qc -> min_rtt + 5000 * 2 ){
	// 	qc -> keep_times = qc -> keep_times + 1;
	// }

}

static void kerdqn_tcp_state(struct sock *sk, u8 new_state)
{
}

static void kerdqn_tcp_cwnd_event(struct sock *sk, enum tcp_ca_event event)
{
}

static void kerdqn_tcp_acked(struct sock *sk, const struct ack_sample *sample)
{
}

static u32 kerdqn_undo_cwnd(struct sock *sk)
{
	return tcp_sk(sk)->snd_cwnd;
}

static struct netlink_kernel_cfg cfg = { 
        .input  = kerdqn_netlink_rcv_msg, 
};  


static struct tcp_congestion_ops kerdqn __read_mostly = {
	.init		= kerdqn_tcp_init,
	.ssthresh	= kerdqn_tcp_recalc_ssthresh,
	.cong_control	= kerdqn_tcp_cong_main,
	.set_state	= kerdqn_tcp_state,
	.undo_cwnd	= kerdqn_undo_cwnd,
	.cwnd_event	= kerdqn_tcp_cwnd_event,
	.pkts_acked     = kerdqn_tcp_acked,
	.owner		= THIS_MODULE,
	.name		= "kerdqn",
};

static int __init kerdqn_register(void)
{
	BUILD_BUG_ON(sizeof(struct kerdqn_bictcp) > ICSK_CA_PRIV_SIZE);

    kerdqn_nlsk = (struct sock *)netlink_kernel_create(&init_net, kerdqn_NETLINK_TEST, &cfg);
    if(kerdqn_nlsk == NULL)
    {   
        printk("netlink_kernel_create error !\n");
        return -1; 
    }   
    printk("netlink_test_init\n");

	return tcp_register_congestion_control(&kerdqn);
}

static void __exit kerdqn_unregister(void)
{
	tcp_unregister_congestion_control(&kerdqn);

    if (kerdqn_nlsk){
        netlink_kernel_release(kerdqn_nlsk); 
        kerdqn_nlsk = NULL;
    }   
    printk("netlink_test_exit!\n");
}

module_init(kerdqn_register);
module_exit(kerdqn_unregister);

MODULE_AUTHOR("xuheng");
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("TCP kerdqn");
MODULE_VERSION("2.3");
    
